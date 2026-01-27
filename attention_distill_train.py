import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5EncoderModel

from distillation.selfattention import GPTClassifier
from distillation.layermap import build_layer_map, attention_distill_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


df = pd.read_csv(r"dataset\Tweets.csv")

df = df[df["sentiment"].isin(["positive", "negative"])].reset_index(drop=True)
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(frac=0.2, random_state=42))
      .reset_index(drop=True)
)

print(df["sentiment"].value_counts())

texts = df["text"].tolist()


teacher_name = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(teacher_name)
teacher = T5EncoderModel.from_pretrained(
    teacher_name,
    output_attentions=True
).to(device)

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False


student = GPTClassifier(
    vocab_size=tokenizer.vocab_size,
    embed_size=400,
    context=312,
    n_heads=8,
    n_layers=4
).to(device)

optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)


loader = DataLoader(
    texts,
    batch_size=1,     
    shuffle=True
)


with torch.no_grad():
    dummy = tokenizer("", return_tensors="pt").input_ids.to(device)
    t_attn = teacher(dummy).attentions
    _, s_attn = student(dummy, return_attn=True)

layer_map = build_layer_map(
    num_teacher_layers=len(t_attn),
    num_student_layers=len(s_attn)
)

print("Layer map:", layer_map)


student.train()

for step, batch_text in enumerate(loader):
    enc = tokenizer(
        batch_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )

    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        teacher_attns = teacher(input_ids).attentions

    _, student_attns = student(input_ids, return_attn=True)

    loss = attention_distill_loss(
        student_attns,
        teacher_attns,
        layer_map
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Attention loss: {loss.item():.6f}")

    if step >= 500:   
        break


os.makedirs("./model", exist_ok=True)

torch.save(
    {
        "model_state_dict": student.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "embed_size": 400,
        "context": 312,
        "n_heads": 8,
        "n_layers": 4,
    },
    "./model/student_attention_distilled.pt"
)

torch.cuda.empty_cache()
print("Student attention-distilled model saved.")
