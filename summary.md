## Clean summary (background + script description)

### Problem setting

We observe a system of similar components. At each discrete, evenly spaced time step, each component produces a binary spike/no-spike event, with a strict global constraint on how many components spike:

- In the original format, exactly 5 out of 50 components spike at every time step.
- In the 12-column format, exactly 2 components spike at every time step.

The data is stored in a growing CSV file as a binary matrix  
[  
Y \in {0,1}^{T \times N},  
]  
with **no header** and **no time/index column**. The sequence length (T) is not fixed (it grows slowly, about **2 new rows per week**).

We assume:

- spike probabilities **oscillate over time** according to an unknown law,
- there are **interrelations** across components (shared/collective dynamics),
- there are **no external covariates**,
- all components have the **same baseline** (no permanently “stronger” components),
- short-term **persistence** is possible (a component may spike 2–3 consecutive steps sometimes).

### Goal

We want one-step-ahead best-guess forecasting: given all observed rows up to time (T), predict which components will spike at time (T+1), while matching the active-count constraint specified on the command line.

---

## Core modeling idea: score-and-rank with a Top-k constraint

A key modeling choice is to not treat components as independent Bernoulli variables. Instead, the model outputs a score for each component, and the prediction is always:

- the Top-k scoring components, where k is supplied on the command line.

This guarantees that the forecast always contains the required number of spikes.

---

## Why we use a GRU (instead of a fixed per-time latent vector)

An earlier idea was to learn a separate latent vector (z_t) for every time step (a trainable latent trajectory). That works when (T) is fixed (e.g., exactly 400), but it becomes awkward when the CSV grows, because the number of latent parameters must grow with (T).

Because (T) is continuously increasing and you want to **retrain from scratch each run**, we use a recurrent sequence model (GRU) whose parameter count does **not** depend on (T). It naturally handles sequences of any length.

---

## Script model (GRU + signed carryover + Top-k output)

### Inputs

At each time step (t), the input is the observed N-dimensional binary vector (Y_t).

### Recurrent hidden state

A **GRU** processes the entire history and produces a hidden state (h_t) that can capture:

- shared oscillatory/latent dynamics,
- cross-component dependencies,
- memory effects.

### One-step score for the next time step

From the hidden state we compute component scores for the next step:

[  
s_{t+1} = \text{Linear}(h_t) + c \cdot Y_t  
]

- `Linear(h_t)` gives a learned score per component based on the latent state.
- `c` is a learned **global carryover strength**. It can be positive when the series tends to persist, or negative when the series tends to rotate away from the previous active set.

Using a signed carryover avoids hard-wiring the model to copy the last row when the observed sequence is mostly anti-persistent.

### Forecast rule (best guess)

The one-step forecast for time (T+1) is:

[  
\hat S_{T+1} = \mathrm{TopK}(s_{T+1})  
]

i.e., pick the k indices with the highest scores.

---

## Training objective (ranking loss aligned with Top-k forecasting)

Since the output is a set/ranking, training is done with a pairwise ranking loss:

- the components that truly spiked at time (t) should have higher scores than components that did not spike.

To keep training efficient, the loss uses negative sampling.

This directly optimizes the behavior we care about: putting the true spikes high in the ranking, so Top-k prediction works well.

---

## Data handling and sanity checks

The script loads the CSV using `pd.read_csv(..., header=None)`, binarizes values using a threshold (> 0.5), and enforces:

- the column width found in the CSV,
- the expected number of ones per row from `--picks`.

---

## Operational usage (retrain-from-scratch each time)

Because the dataset grows slowly (~2 rows/week), retraining from scratch is practical:

1. Append new rows to your CSV file.
2. Run the script.
3. The script:
   - loads all available rows (whatever (T) is),
   - trains a new model from random initialization (optionally with fixed seeds for reproducibility),
   - prints the predicted indices for the next time step (T+1).

### Command-line configuration

The script accepts the data path as a positional argument and takes the row constraint from the CLI:

```bash
python binary_oscillator.py sample_data_12.csv --picks 2
```

For the original 50-column data, use:

```bash
python binary_oscillator.py data.csv --picks 5
```

The number of columns is read directly from the CSV file, so it does not need to appear in the command.

### Sample file

A small verification file is included as `sample_data_12.csv`. It contains 8 rows, 12 columns, and exactly two ones per row.

---

## Output interpretation

The output is a deterministic best-guess one-step forecast: the component indices predicted to spike next, under the active-count rule implied by the input format. The script is intended for forecasting (ranking), not for generating full probabilistic simulations of the future.





## **Claudes interpretation**

## Analysis of `binary_oscillator.py`

This script is a **binary sequence forecaster** — it learns patterns from historical binary (on/off) data and predicts which entries will be "active" in the next time step. Here's a breakdown:

---

### 🎯 Core Purpose

Given a CSV of historical rows where **exactly `--picks` entries per row are active (1.0)** and the rest are inactive (0.0), it trains a recurrent neural network to predict **which `--picks` entries will be active next**.

---

### 🧱 Components

#### 1. **Data Loading — `load_Y()`**

- Reads a CSV file (no header) of numerical values
- Binarizes each value: `> 0.5 → 1.0`, else `0.0`
- **Validates** that every row sums to exactly `--picks` active entries, raising an error otherwise
- Returns a PyTorch tensor of shape `[T, N]` (T = time steps, N = number of columns)

#### 2. **Model — `TopKGRU`**

A lightweight recurrent model with three parts:

| Layer                      | Role                                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `nn.GRU`                   | Encodes temporal sequence patterns (hidden size = 32)                                                       |
| `nn.Linear`                | Projects hidden state → scores for each of the N entries                                                    |
| `carryover` (scalar param) | Learned weight that blends the previous active set into the next prediction — a persistence/momentum signal |

The score for the next step is:

```
scores = Linear(h_prev) + carryover × y_prev
```

#### 3. **Loss Function — `ranking_loss()`**

Uses a **pairwise ranking loss** (softplus margin):

- For each positive (active) entry, it samples `neg_per_pos=20` random negative entries
- Penalizes cases where a negative scores higher than a positive
- Effectively trains the model to **rank active entries above inactive ones**

This is more appropriate than cross-entropy here because the goal is relative ordering (top-K selection), not calibrated probabilities.

#### 4. **Training — `train_from_scratch()`**

- Uses **AdamW** optimizer with weight decay (L2 regularization)
- Runs the full sequence through the GRU in one batch, then steps through each time transition `t-1 → t` to accumulate ranking loss
- Trains for **120 epochs**, printing progress every 20 epochs along with the learned `carryover` value

#### 5. **Forecasting — `forecast_one_step()`**

- Passes the full historical sequence through the GRU
- Uses the **last hidden state** + **last row** to score all N entries
- Returns the **top-K indices** (both 0-based and 1-based) as the forecast

---

### 🔁 Execution Flow

```
Parse args (data_path, --picks)
       ↓
Load & validate CSV → binary tensor Y [T, N]
       ↓
Build TopKGRU model (N inputs, 32 hidden)
       ↓
Train 120 epochs with pairwise ranking loss
       ↓
Forecast: output top-K indices for the next step
```

---

### 💡 Key Design Choices & Observations

| Aspect                 | Detail                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| **Determinism**        | Seeds `numpy` and `torch` with `0` for reproducible retrains                                            |
| **Device support**     | Automatically uses CUDA if available                                                                    |
| **`carryover` param**  | Captures persistence — if recently active entries tend to stay active, this will learn a positive value |
| **No softmax/sigmoid** | Raw scores are used; only relative ranking matters                                                      |
| **Single-step ahead**  | Only predicts one step into the future, not multi-step                                                  |

---

### 🚀 Typical Use Case

This script is well-suited for **lottery-style or sports pick prediction**, binary schedule/rotation forecasting, or any domain where exactly K out of N binary slots are active per time step and you want to predict the next draw based on history.

**Example usage:**

```bash
python binary_oscillator.py history.csv --picks 6
```
