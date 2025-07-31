try:
    from ulab import nupy as np
except ImportError:
    import numpy as np

from .xtremepy import XtremePy


class MicroXtremePy:
    def __init__(self, window=100, gamma=1.5, nxtrem=4, moutliers=5, delta=0.99, sma=5):
        self.window = window
        self.gamma = gamma
        self.nxtrem = nxtrem
        self.moutliers = moutliers
        self.delta = delta
        self.sma = sma

        self.model_pos = XtremePy(window=self.window, gamma=self.gamma,
                                  nxtrem=self.nxtrem, moutliers=self.moutliers,
                                  delta=self.delta, sma=self.sma)

        self.model_neg = XtremePy(window=self.window, gamma=self.gamma,
                                  nxtrem=self.nxtrem, moutliers=self.moutliers,
                                  delta=self.delta, sma=self.sma)
        return self

    def predict(self, X):
        preds = []
        for xi in X:
            out_pos = self.model_pos.update(xi)
            out_neg = self.model_neg.update(-xi)

            is_pos_outlier = out_pos["outlier"]
            is_neg_outlier = out_neg["outlier"]

            preds.append(1 if is_pos_outlier or is_neg_outlier else 0)
        return np.array(preds)
