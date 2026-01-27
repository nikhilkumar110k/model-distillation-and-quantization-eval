class AffineQuantizer:
    def __init__(self, num_bits=8):
        self.qmin = 0
        self.qmax = 2**num_bits - 1
        self.scale = None
        self.zero_point = None

    def calculate_qparams(self, rmin, rmax):
        eps = 1e-8
        scale = (rmax - rmin) / (self.qmax - self.qmin + eps)

        zero_point = self.qmin - rmin / scale
        zero_point = int(round(zero_point))
        zero_point = max(self.qmin, min(self.qmax, zero_point))

        self.scale = scale
        self.zero_point = zero_point

        return scale, zero_point
