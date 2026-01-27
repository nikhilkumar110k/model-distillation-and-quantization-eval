import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import kagglehub
from kagglehub import KaggleDatasetAdapter
from distillation.selfattention import GPTClassifier
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
import pandas as pd 


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

df = pd.read_csv(r"dataset\Tweets.csv")


df = df[df["sentiment"].isin(["positive", "negative"])].reset_index(drop=True)

label_map = {"negative": 0, "positive": 1}

df["label"] = df["sentiment"].map(label_map)

df = (
    df.groupby(as_index=False, by="label", group_keys=False)
      .apply(lambda x: x.sample(frac=0.2, random_state=42))
      .reset_index(drop=True)
)

print(df.head())
print(df["sentiment"].value_counts())

teacher_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(teacher_name)

class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )

            return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "label": torch.tensor(self.labels[idx], dtype=torch.long)
    }

        


dataset = TwitterDataset(
    df["text"].tolist(),
    df["label"].tolist(),
    tokenizer
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)




teacher = AutoModelForSequenceClassification.from_pretrained(
    teacher_name
).to(device)

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False


ckpt = torch.load(
    "./model/student_attention_distilled.pt",
    map_location=device
)

student = GPTClassifier(
    vocab_size=ckpt["vocab_size"],
    embed_size=ckpt["embed_size"],
    context=ckpt["context"],
    n_heads=ckpt["n_heads"],
    n_layers=ckpt["n_layers"],
).to(device)

student.load_state_dict(ckpt["model_state_dict"])

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)


inputs = tokenizer(
    df["text"].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
).input_ids.to(device)


def kd_loss(student_logits, teacher_logits, T=3.0):
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)

ce_loss = torch.nn.CrossEntropyLoss()
alpha = 0.4

student.train()

for epoch in range(8):
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():

            
            teacher_logits = teacher(
    input_ids=input_ids,
    attention_mask=attention_mask
).logits

        student_logits = student(input_ids)

        loss_kd = kd_loss(student_logits, teacher_logits, T=3.0)
        loss_ce = ce_loss(student_logits, labels)

        loss = alpha * loss_kd + (1 - alpha) * loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Avg loss: {total_loss / len(loader):.4f}")



os.makedirs("./model", exist_ok=True)

torch.save(
    {
        "model_state_dict": student.state_dict(),
        "vocab_size": ckpt["vocab_size"],
        "embed_size": ckpt["embed_size"],
        "context": ckpt["context"],
        "n_heads": ckpt["n_heads"],
        "n_layers": ckpt["n_layers"],
    },
    "./model/student_logits_distilled.pt",
)
torch.cuda.empty_cache()
print("Final student model saved.")
