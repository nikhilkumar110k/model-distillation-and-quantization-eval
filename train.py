import torch
from transformers import T5Tokenizer, T5EncoderModel

from selfattention import GPTClassifier
from layermap import build_layer_map, attention_distill_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
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

text = ["this movie was surprisingly good"]
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
).input_ids.to(device)

with torch.no_grad():
    teacher_out = teacher(inputs)
    teacher_attns = teacher_out.attentions

_, student_attns = student(inputs, return_attn=True)

print("Teacher layers:", len(teacher_attns))
print("Student layers:", len(student_attns))
print("Teacher attn shape:", teacher_attns[0].shape)
print("Student attn shape:", student_attns[0].shape)


