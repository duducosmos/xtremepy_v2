try:
    from ulab import numpy as np
except ImportError:
    import numpy as np

from typing import Literal
import ujson as json
# funciona com MicroPython + WiFi (ESP32, RP2040 W)
import urequests as requests

from .xtremepy import XtremePy
from .microxtremedetector import MicroXtremePy


class FactoryXtremePy:
    def __init__(self, config_url=None):
        self.config_url = config_url
        self.params = {
            "window": 100,
            "gamma": 1.5,
            "nxtrem": 4,
            "moutliers": 5,
            "delta": 0.99,
            "sma": 5
        }
        self.model = self._build_model()

    def _build_model(self) -> MicroXtremePy:
        return MicroXtremePy(**self.params)

    def set_params(self, **kwargs) -> Literal[True]:
        # Atualiza os parâmetros se fornecidos
        for key in self.params:
            if key in kwargs:
                self.params[key] = kwargs[key]
        self.model = self._build_model()
        return True

    def get_model(self) -> MicroXtremePy:
        return self.model

    def check_for_remote_config(self) -> bool:
        if not self.config_url:
            return False
        try:
            res = requests.get(self.config_url)
            if res.status_code == 200:
                config = res.json()
                self.set_params(**config)
                print("[XtremePy] Remote config loaded.")
                return True
        except Exception as e:
            print("[XtremePy] Failed to load remote config:", e)
        return False
