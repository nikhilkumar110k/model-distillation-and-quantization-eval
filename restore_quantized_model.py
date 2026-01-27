import torch
from distillation.selfattention import GPTClassifier


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
qstate = torch.load("./model/student_int8.pt", map_location="cpu")

student = GPTClassifier(
    vocab_size=32000,
    embed_size=400,
    context=312,
    n_heads=8,
    n_layers=4,
)

state_dict = {}

for name, info in qstate.items():
    if "qweight" in info:
        w = info["qweight"]
        scale = info["scale"]
        zp = info["zero_point"]

        state_dict[name] = scale * (w.float() - zp)
    else:
        state_dict[name] = info["fp"]

student.load_state_dict(state_dict, strict=True)
student.eval()

torch.save(student.state_dict(), "./model/student_dequantized.pt")
print("Dequantized model restored")


ckpt = torch.load("./model/student_dequantized.pt", map_location=device)

student = GPTClassifier(
    vocab_size=32000,
    embed_size=400,
    context=312,
    n_heads=8,
    n_layers=4,
).to(device)

student.load_state_dict(ckpt)
student.eval()
