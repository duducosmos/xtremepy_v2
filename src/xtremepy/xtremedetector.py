import json
from typing import Any, Self
from sklearn.base import BaseEstimator
import numpy as np
import optuna
from sklearn.metrics import f1_score

from .xtremepy import XtremePy


class XtremeDetector(BaseEstimator):
    def __init__(self, window=100, gamma=1.5,  nxtrem=4, moutliers=5, delta=0.99, sma=5):
        self.window = window
        self.gamma = gamma
        self.nxtrem = nxtrem
        self.moutliers = moutliers
        self.delta = delta
        self.sma = sma

    def fit(self, X, y=None) -> Self:
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

            is_pos_outlier = out_pos and out_pos["outlier"]
            is_neg_outlier = out_neg and out_neg["outlier"]

            preds.append(1 if is_pos_outlier or is_neg_outlier else 0)
        return np.array(preds)

    def tune(self, X, y, n_trials=50, timeout=None, verbose=True) -> Self:
        def objective(trial):
            params = {
                "window": trial.suggest_int("window", 10, 300),
                "gamma": trial.suggest_float("gamma", 0.5, 3.5),
                "nxtrem": trial.suggest_int("nxtrem", 4, 30),
                "moutliers": trial.suggest_int("moutliers", 1, 20),
                "delta": trial.suggest_float("delta", 0.90, 0.999),
                "sma": trial.suggest_int("sma", 1, 20),
            }

            model_pos = XtremePy(**params)
            model_neg = XtremePy(**params)

            y_pred = []
            for xi in X:
                out_pos = model_pos.update(xi)
                out_neg = model_neg.update(-xi)
                out = (out_pos and out_pos["outlier"]) or (
                    out_neg and out_neg["outlier"])
                y_pred.append(1 if out else 0)
            return f1_score(y, np.array(y_pred).astype(int))
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        if verbose:
            print("Melhores hiperparâmetros encontrados:")
            for k, v in study.best_params.items():
                print(f"{k}: {v}")
            print(f"Melhor F1-score: {study.best_value:.4f}")

        for k, v in study.best_params.items():
            setattr(self, k, v)

        self.fit(X)  # Reinstancia com os melhores parâmetros
        return self

    def save_params_to_json(self, params, filepath="xtremepy_params.json"):
        with open(filepath, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Parameters saved to {filepath}")

    def run_tuning_and_export(self, X, y, n_trials=50, timeout=None, save_path="xtremepy_params.json") -> dict[str, Any]:
        detector = XtremeDetector()
        detector.tune(X, y, n_trials=n_trials, timeout=timeout, verbose=True)

        # Extrair parâmetros atuais (melhores encontrados)
        best_params = {
            "window": detector.window,
            "gamma": detector.gamma,
            "nxtrem": detector.nxtrem,
            "moutliers": detector.moutliers,
            "delta": detector.delta,
            "sma": detector.sma,
        }

        self.save_params_to_json(best_params, save_path)
        return best_params
