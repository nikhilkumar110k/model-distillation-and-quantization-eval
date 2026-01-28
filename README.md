Model Distillation & Quantization (From Scratch)

This repository is a from-scratch, implementation of Transformer model distillation, focusing on attention distillation, logits (KD) distillation, and preparation for quantization — without relying on high-level distillation frameworks.

The goal of this project is not to train a state-of-the-art classifier out of the box, but to understand, implement, and validate the mechanics of distillation end‑to‑end.

🔍 What This Project Demonstrates

Custom encoder-only Transformer student (4 layers)

T5-style relative positional attention

Attention distillation (MSE on attention maps)

Logits distillation (KL divergence with temperature)

Mean-based teacher → student layer mapping

Explicit control over temperature smoothing

Clean save / load / evaluation pipeline

This mirrors techniques used in:

TinyBERT

MiniLM

DistilBERT (variants)


🧩 Student Model Overview

Encoder-only Transformer

4 layers, 8 heads

T5-style relative position bucket bias

Classification head (Linear → 2 logits)

