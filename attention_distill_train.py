import torch
from transformers import T5Tokenizer, T5EncoderModel
import os
from distillation.selfattention import GPTClassifier
from distillation.layermap import build_layer_map, attention_distill_loss
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

teacher_name = "google/flan-t5-large"

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


text = ["this movie was surprisingly good"]
inputs = tokenizer(
    df["text"].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
).input_ids.to(device)


with torch.no_grad():
    teacher_attns = teacher(inputs).attentions

_, student_attns = student(inputs, return_attn=True)

layer_map = build_layer_map(
    num_teacher_layers=len(teacher_attns),
    num_student_layers=len(student_attns)
)

print("Layer map:", layer_map)


student.train()

for step in range(10):
    with torch.no_grad():
        teacher_attns = teacher(inputs).attentions

    _, student_attns = student(inputs, return_attn=True)

    loss = attention_distill_loss(
        student_attns,
        teacher_attns,
        layer_map
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step} | Attention loss: {loss.item():.6f}")


os.makedirs("./model", exist_ok=True)

save_path = "./model/student_attention_distilled.pt"

torch.save(
    {
        "model_state_dict": student.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "embed_size": 400,
        "context": 312,
        "n_heads": 8,
        "n_layers": 4,
    },
    save_path
)
torch.cuda.empty_cache()
print(f"Student model saved to {save_path}")