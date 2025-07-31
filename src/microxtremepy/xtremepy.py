try:
    from ulab import numpy as np
except ImportError:
    import numpy as np


class XtremePy:
    def __init__(self, window=270, gamma=1.45, taverage=700, nxtrem=11, moutliers=1, delta=0.99, sma=2):
        self.window = window
        self.gamma = gamma
        self.taverage = taverage
        self.nxtrem = nxtrem
        self.moutliers = moutliers
        self.delta = delta
        self.sma = sma
        self.buffer = []
        self.sma_buffer = []
        self.n = 0
        self.zq = 0

    def update(self, value):
        self.n += 1
        self.buffer.append(value)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

        self.sma_buffer.append(value)
        if len(self.sma_buffer) > self.sma:
            self.sma_buffer.pop(0)
        smoothed = np.mean(self.sma_buffer)

        if len(self.buffer) < self.window:
            return {"outlier": 0, "regime_change": 0}

        sorted_vals = sorted(self.buffer, reverse=True)
        threshold_index = min(self.nxtrem - 1, len(sorted_vals) - 1)
        threshold = sorted_vals[threshold_index]
        outlier = int(smoothed > threshold * self.gamma)

        self.zq = self.delta * self.zq + (1 - self.delta) * outlier
        regime_change = int(self.zq > self.moutliers)
        if regime_change:
            self.zq = 0  # reset

        return {"outlier": outlier, "regime_change": regime_change}
