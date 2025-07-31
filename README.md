# XtremePy

**XtremePy** is an efficient online anomaly detection library inspired by the principles of **Extreme Value Theory (EVT)**. It is designed for real-time signal processing with low memory and computational overhead — ideal for embedded systems, streaming data, and non-stationary time series.

---

## 🔬 Theoretical Foundation

XtremePy is grounded in the idea that **extreme observations (maxima or minima)** in a time series carry the most information about anomalous behavior. Traditional EVT approaches rely on:

- Generalized Extreme Value (GEV) distributions for block maxima,
- Generalized Pareto Distributions (GPD) over thresholds (Peaks Over Threshold - POT).

However, such methods are often computationally intensive. XtremePy offers a **non-parametric, rank-based approximation of EVT** that captures its core intuition: **rare, high-magnitude events are statistically distinguishable and often signal regime shifts or anomalies**.

---

## ⚙️ Algorithm Overview

Each incoming observation is processed as follows:

1. **Sliding Window Buffer**  
   A rolling buffer of the `N` most recent samples is maintained.

2. **Rank-based Threshold**  
   The buffer is sorted in descending order. The `k`-th most extreme value (max or min) defines a threshold, scaled by a sensitivity factor `γ`.

3. **Smoothing (Optional)**  
   A simple moving average (`sma`) can be applied to incoming data to reduce short-term volatility.

4. **Outlier Detection**  
   An observation is flagged as an outlier if it exceeds the threshold.

5. **Regime Change Detection**  
   A cumulative score `zq`, updated using an exponential moving average with decay `δ`, tracks the frequency of outliers. A regime change is detected if this score exceeds a configurable threshold.

---

## 📦 Installation

```bash
pip install xtremepy
````

> Or clone the repository:

```bash
git clone https://github.com/your-org/xtremepy.git
cd xtremepy
pip install -e .
```

---

## 🔧 API

```python
from xtremepy import AMSystem

model = AMSystem(
    window=270,
    gamma=1.45,
    taverage=700,
    nxtrem=11,
    moutliers=1,
    delta=0.99,
    sma=2
)

for xi in stream:
    result = model.update(xi)
    if result["outlier"]:
        print("Anomaly detected!")
```

---

## 📊 Hyperparameter Optimization

Use `optuna` to tune model parameters based on labeled data:

```python
from xtremepy import AMSystemDetector

detector = AMSystemDetector()
detector.tune(X_train, y_train, n_trials=50)
y_pred = detector.predict(X_test)
```

---

## 📈 Performance

Supports comparison with other methods:

* Isolation Forest
* Prophet residuals
* Z-score thresholding

XtremePy performs well on synthetic and real-world time series, particularly in **low-latency environments**.

---

## 🧪 Example

```python
from xtremepy.utils import gen_data
from xtremepy import AMSystemDetector
from sklearn.model_selection import train_test_split

X, y = gen_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = AMSystemDetector()
model.tune(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 📚 Citation

If you use XtremePy in academic work:

> **Pereira, E.** (2025). *XtremePy: A Lightweight EVT-Based Anomaly Detection Framework for Streaming Data*. INPI Registration No. XXXXXXX.

---

## 🛠 Parameters

| Parameter   | Description                         |
| ----------- | ----------------------------------- |
| `window`    | Size of rolling buffer              |
| `gamma`     | Threshold scaling factor            |
| `taverage`  | Not used (reserved for future use)  |
| `nxtrem`    | Number of extreme values considered |
| `moutliers` | Regime change threshold             |
| `delta`     | Forgetting factor for regime change |
| `sma`       | Size of moving average window       |

---

## ✅ License

MIT License

---

## 🤝 Contributions

Feel free to open issues or pull requests. Suggestions for additional anomaly detection features or benchmarks are welcome!


