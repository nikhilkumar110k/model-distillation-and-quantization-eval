import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from distillation.selfattention import GPTClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


teacher_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(teacher_name)

teacher = AutoModelForSequenceClassification.from_pretrained(
    teacher_name
).to(device)

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False


ckpt = torch.load(
    "./model/student_int8.pt",
    map_location=device
)

student = GPTClassifier(
    vocab_size=12800,
    embed_size=128,
    context=128,
    n_heads=6,
    n_layers=4,
).to(device)

student.load_state_dict(ckpt["model_state_dict"])
student.eval()

texts = [
    "this movie was amazing and I loved it",
    "this was the worst experience of my life",
    "absolutely fantastic acting and story",
    "i hate this product so much",
    "it was okay, not great but not terrible",
]

enc = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
)

input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)


with torch.no_grad():
    teacher_logits = teacher(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).logits

    student_logits = student(input_ids)

    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)


for i, text in enumerate(texts):
    print("=" * 70)
    print(text)

    print(
        f"Teacher  → Negative: {teacher_probs[i,0]:.3f} | "
        f"Positive: {teacher_probs[i,1]:.3f}"
    )

    print(
        f"Student  → Negative: {student_probs[i,0]:.3f} | "
        f"Positive: {student_probs[i,1]:.3f}"
    )
