import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel

from distillation.selfattention import GPTClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


teacher_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(teacher_name)

teacher = T5EncoderModel.from_pretrained(
    teacher_name,
    output_attentions=False
).to(device)

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

teacher_classifier = torch.nn.Linear(
    teacher.config.d_model, 2
).to(device)

teacher_classifier.eval()
for p in teacher_classifier.parameters():
    p.requires_grad = False

ckpt = torch.load("./model/student_logits_distilled.pt", map_location=device)

student = GPTClassifier(
    vocab_size=ckpt["vocab_size"],
    embed_size=ckpt["embed_size"],
    context=ckpt["context"],
    n_heads=ckpt["n_heads"],
    n_layers=ckpt["n_layers"],
).to(device)

student.load_state_dict(ckpt["model_state_dict"])
student.eval()

texts = [
    "this movie was amazing and I loved it",
    "this was the worst experience of my life",
]
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
).input_ids.to(device)

with torch.no_grad():
    teacher_hidden = teacher(inputs).last_hidden_state
    teacher_logits = teacher_classifier(teacher_hidden[:, -1, :])
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    student_logits = student(inputs)
    student_probs = F.softmax(student_logits, dim=-1)

for i, text in enumerate(texts):
    print("=" * 60)
    print(text)
    print("Teacher probs :", teacher_probs[i].cpu().numpy())
    print("Student probs :", student_probs[i].cpu().numpy())
