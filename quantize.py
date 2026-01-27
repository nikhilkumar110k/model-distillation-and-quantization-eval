import torch
import torch.nn as nn
from  quantizer.affinequantize import AffineQuantizer
from quantizer.observer_quantizer import MinMaxObserver



def quantize_affine(x, scale, zero_point, qmin, qmax):
    q = torch.round(x / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.int32)

def dequantize_affine(q, scale, zero_point):
    return scale * (q.float() - zero_point)


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_fp = nn.Parameter(
            torch.randn(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.register_buffer("weight_q", None)
        self.scale = None
        self.zero_point = None

    def calibrate(self):
        observer = MinMaxObserver()
        observer.observe(self.weight_fp.data)

        rmin, rmax = observer.get_minmax()

        quantizer = AffineQuantizer(num_bits=8)
        scale, zero_point = quantizer.calculate_qparams(rmin, rmax)

        self.scale = scale
        self.zero_point = zero_point

        self.weight_q = quantize_affine(
            self.weight_fp.data,
            scale,
            zero_point,
            quantizer.qmin,
            quantizer.qmax
        )

    def forward(self, x):
        
        acc = torch.matmul(
            x,
            (self.weight_q - self.zero_point).t().float()
        )

        out = acc * self.scale + self.bias
        return out
