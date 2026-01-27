import torch
from distillation.selfattention import GPTClassifier
from quantizer.quantize import quantize_affine, calculate_qparams

device = "cpu"  

ckpt = torch.load(
    "./model/student_logits_distilled.pt",
    map_location="cpu"
)

student = GPTClassifier(
    vocab_size=ckpt["vocab_size"],
    embed_size=ckpt["embed_size"],
    context=ckpt["context"],
    n_heads=ckpt["n_heads"],
    n_layers=ckpt["n_layers"]
)

student.load_state_dict(ckpt["model_state_dict"])
student.eval()

quant_state = {}

for name, param in student.named_parameters():
    if param.ndim >= 2:  

        scale,zp=calculate_qparams(param.data)
        q = quantize_affine(param.data, scale, zp)

        quant_state[name] = {
            "qweight": q,
            "scale": scale,
            "zero_point": zp,
            "shape": param.shape
        }
    else:
        quant_state[name] = {
            "fp": param.data
        }

torch.save(quant_state, "./model/student_int8.pt")

print("Quantized model saved")
