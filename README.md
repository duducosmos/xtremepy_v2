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



### Mathematical Basis of the XtremePy Algorithm Inspired by Extreme Value Theory (EVT)

The `XtremePy` algorithm implements a sliding-window based outlier detection mechanism grounded in concepts from Extreme Value Theory (EVT), which studies the statistical behavior of the maximum or minimum values in large samples.

1. **Sliding Window and Extreme Order Statistics**

   Given a time series $\{x_t\}$, at each time step $t$, the algorithm maintains a sliding window $W_t = \{x_{t-w+1}, \ldots, x_t\}$ of fixed size $w$ (parameter `window`). This window represents the most recent observations, assumed to be representative of the "normal" data distribution.

   From $W_t$, the algorithm extracts the $k$-th largest value $X_{(k)}$, where $k$ corresponds to the `nxtrem` parameter (number of extremes considered). This order statistic acts as a dynamic threshold that adapts to recent data extremes:

   $$
   X_{(k)} = \text{k-th largest value in } W_t
   $$

2. **Smoothing and Thresholding**

   To reduce noise and volatility, a short-term simple moving average (SMA) of recent observations is computed over a smaller window $s$ (parameter `sma`):

   $$
   \bar{x}_t = \frac{1}{s} \sum_{i=t-s+1}^{t} x_i
   $$

   An observation at time $t$ is flagged as an outlier if the smoothed value exceeds a scaled threshold:

   $$
   \text{outlier}_t = \mathbb{I} \left( \bar{x}_t > \gamma \cdot X_{(k)} \right)
   $$

   where $\gamma > 1$ (parameter `gamma`) inflates the threshold to control the sensitivity and reduce false positives. The indicator function $\mathbb{I}$ returns 1 if the condition is true, otherwise 0.

3. **Exponential Moving Average for Outlier Persistence**

   To track the persistence or accumulation of outliers over time, the algorithm uses an exponentially weighted moving average (EWMA) of recent outlier flags:

   $$
   z_t = \delta \cdot z_{t-1} + (1 - \delta) \cdot \text{outlier}_t
   $$

   Here, $\delta \in [0,1)$ (parameter `delta`) controls the memory of past outliers. A larger $\delta$ emphasizes older data, smoothing short bursts of outliers, while a smaller $\delta$ reacts quickly to recent anomalies.

4. **Regime Change Detection**

   When the accumulated outlier measure $z_t$ crosses a predefined threshold $m$ (parameter `moutliers`), the algorithm flags a regime change, indicating a significant shift in the underlying data distribution or environment:

   $$
   \text{regime\_change}_t = \mathbb{I}(z_t > m)
   $$

   Upon detecting a regime change, $z_t$ is reset to zero to start fresh monitoring, preventing continuous triggering from the same anomalous period.


### Technical Performance Analysis of Anomaly Detection Algorithms

The table below summarizes the comparative performance metrics of four anomaly detection methods evaluated on the same dataset:

| Method            | Precision | Recall | F1-score |
| ----------------- | --------- | ------ | -------- |
| Isolation Forest  | 0.97      | 1.00   | 0.98     |
| Prophet Residuals | 1.00      | 1.00   | 1.00     |
| Z-score           | 0.94      | 1.00   | 0.97     |
| **XtremePy**      | 0.83      | 1.00   | 0.91     |

**Interpretation:**

* **Prophet Residuals** achieved ideal performance with perfect precision, recall, and F1-score, indicating excellent anomaly characterization with zero false positives or false negatives.
* **Isolation Forest** demonstrates robust anomaly detection capabilities, with a high precision of 0.97 and perfect recall, resulting in a strong F1-score of 0.98. This indicates that it accurately identifies most anomalies with few false alarms.
* The **Z-score** method also maintains strong recall, detecting all anomalies, but shows a modest decrease in precision (0.94), implying some susceptibility to false positives.
* The **XtremePy** algorithm maintains perfect recall (1.00), ensuring no anomalies are missed; however, it has a comparatively lower precision (0.83), indicating a higher false positive rate relative to the other methods. This trade-off suggests that XtremePy is more conservative in its detection, prioritizing sensitivity over specificity.

**Conclusion:**

XtremePy, while highly sensitive, may benefit from further calibration or hybrid approaches to reduce false positive rates without compromising its anomaly detection capability. Its foundation on Extreme Value Theory provides a strong theoretical basis, but additional tuning or complementary filtering could improve precision to levels comparable with established methods like Isolation Forest or Prophet Residuals.

