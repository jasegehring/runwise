# Runwise Examples

## Example: Quick Health Check

**User asks:** "How is training going?"

```bash
$ runwise latest

Run: 2024-01-15_lr-sweep-001 (running)
Step: 45000 | Epoch: 12 | Runtime: 4h 23m

Metrics:
  loss:      ▆▅▄▃▂▂▁▁  0.892 → 0.234 ↓
  val_loss:  ▆▅▄▃▃▂▂▂  0.901 → 0.289 ↓
  accuracy:  ▁▂▃▄▅▆▇█  0.65 → 0.91  ↑
  lr:        █▇▆▅▄▃▂▁  1e-3 → 1e-5  ↓

No anomalies detected.
```

**Interpretation:** Training is healthy - loss decreasing, accuracy increasing, no anomalies.

---

## Example: Detecting Overfitting

**User asks:** "Is my model overfitting?"

```bash
$ runwise latest -k loss,val_loss

Run: experiment-42 (running)
Step: 80000

Metrics:
  loss:      ▅▄▃▂▁▁▁▁  0.45 → 0.08  ↓
  val_loss:  ▄▃▃▃▄▅▆▇  0.52 → 0.89  ↑

⚠ ANOMALIES DETECTED:
  • Overfitting: val_loss/loss ratio 11.1x (threshold: 2.0x) at step 80000
```

**Interpretation:** Yes, overfitting detected. Train loss keeps dropping but val loss is rising.

---

## Example: Comparing Runs

**User asks:** "Compare my baseline to the new run"

```bash
$ runwise list
ID        Name              Status    Steps   val_loss
abc123    baseline-v1       finished  100000  0.312
def456    new-augmentation  finished  100000  0.287

$ runwise compare abc123 def456 -f val -t 5

Comparing: baseline-v1 vs new-augmentation

Metric          baseline-v1    new-augmentation    Delta
val_loss        0.312          0.287               -8.0% ✓
val_accuracy    0.891          0.912               +2.4%
val_f1          0.867          0.901               +3.9%
```

**Interpretation:** New augmentation improves val_loss by 8% and other val metrics.

---

## Example: Step-Matched Comparison

**User asks:** "Compare runs at 50k steps" (important for curriculum learning)

```bash
$ runwise compare run1@50000 run2@50000 -f val

Comparing at step 50000:

Metric          run1@50000    run2@50000    Delta
val_loss        0.412         0.389         -5.6% ✓
val_accuracy    0.823         0.841         +2.2%
```

---

## Example: Discovering Metrics

**User asks:** "What metrics are being logged?"

```bash
$ runwise keys

Available metrics in latest run (abc123):
  Training: loss, accuracy, lr, gradient_norm
  Validation: val_loss, val_accuracy, val_f1, val_precision, val_recall
  System: throughput, gpu_memory, gpu_util
```

---

## Example: Getting Raw History

**User asks:** "Show me the loss values"

```bash
$ runwise history -k loss,val_loss --samples 20

step,loss,val_loss
0,2.341,2.356
5000,0.892,0.901
10000,0.534,0.567
15000,0.412,0.445
20000,0.334,0.378
...
```

---

## Example: Stability Analysis

**User asks:** "Is training stable?"

```bash
$ runwise stability -k loss,val_loss

Stability Analysis (window=100):

Metric      Mean    Std Dev    Max Std    Stable?
loss        0.234   0.012      0.034      ✓ Yes
val_loss    0.289   0.045      0.123      ⚠ Moderate variance
```

---

## Example: Live Status

**User asks:** "What's happening right now?"

```bash
$ runwise live

Live Status (from output.log):
  Run ID: abc123
  Step: 45023
  Epoch: 12/50
  Loss: 0.234
  LR: 2.5e-5
  Throughput: 1234 samples/sec
  ETA: 2h 15m
```

---

## Example: Local JSONL Logs

**User asks:** "Analyze my local training log"

```bash
$ runwise local
Available logs in logs/:
  training_2024-01-15.jsonl  (45000 lines)
  training_2024-01-14.jsonl  (100000 lines)

$ runwise local training_2024-01-15.jsonl --keys
Available keys: step, loss, val_loss, accuracy, lr

$ runwise local training_2024-01-15.jsonl --stats -k loss,val_loss
Metric      Min      Max      Mean     Final
loss        0.089    2.341    0.445    0.234
val_loss    0.156    2.356    0.512    0.289
```
