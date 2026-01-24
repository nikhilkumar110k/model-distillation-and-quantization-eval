import torch
import torch.nn.functional as F

def build_layer_map(num_teacher_layers, num_student_layers):
    band = num_teacher_layers // num_student_layers
    layer_map = {}

    for s_idx in range(num_student_layers):
        start = s_idx * band
        end = start + band
        if s_idx == num_student_layers - 1:
            end = num_teacher_layers
        layer_map[s_idx] = list(range(start, end))

    return layer_map

def attention_distill_loss(student_attns, teacher_attns, layer_map):
    loss = 0.0

    for s_idx, t_idxs in layer_map.items():
        s_attn = student_attns[s_idx].mean(dim=1)  

        t_stack = []
        for t_idx in t_idxs:
            t_stack.append(teacher_attns[t_idx].mean(dim=1))

        t_attn = torch.stack(t_stack).mean(dim=0)  

        loss += F.mse_loss(s_attn, t_attn)

    return loss / len(layer_map)


