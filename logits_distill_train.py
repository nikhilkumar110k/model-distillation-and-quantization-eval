import os
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel

from distillation.selfattention import GPTClassifier
from distillation.layermap import build_layer_map, attention_distill_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


teacher_name = "google/flan-t5-large"

tokenizer = T5Tokenizer.from_pretrained(teacher_name)
teacher = T5EncoderModel.from_pretrained(
    teacher_name,
    output_attentions=True
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


texts = ["this movie was surprisingly good"]
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
).input_ids.to(device)


with torch.no_grad():
    teacher_attns = teacher(inputs).attentions

_, student_attns = student(inputs, return_attn=True)

layer_map = build_layer_map(
    num_teacher_layers=len(teacher_attns),
    num_student_layers=len(student_attns),
)

print("Layer map:", layer_map)

def kd_loss(student_logits, teacher_logits, T=4.0):
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


student.train()

for step in range(50):
    with torch.no_grad():
        teacher_out = teacher(inputs)
        teacher_hidden = teacher_out.last_hidden_state
        teacher_logits = teacher_classifier(teacher_hidden[:, -1, :])
        teacher_attns = teacher_out.attentions

    student_logits, student_attns = student(inputs, return_attn=True)

    loss_kd = kd_loss(student_logits, teacher_logits, T=4.0)
    loss_attn = attention_distill_loss(
        student_attns,
        teacher_attns,
        layer_map
    )

    loss = loss_kd + 0.5 * loss_attn

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"Step {step:02d} | KD: {loss_kd.item():.4f} | Attn: {loss_attn.item():.4f}"
    )

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

print("Final student model saved.")
