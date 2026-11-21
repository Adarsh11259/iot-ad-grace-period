# Gap analysis results

Random Forest, reproducing the authors' own method (`scripts/gap_analysis.py`).

| Experiment | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| A) Pooled random 75/25 split (authors' method) | 0.966 | 0.921 | 0.971 | 0.945 |
| B) Held-out-device split (6 devices never seen in training) | 0.908 | 0.836 | 0.977 | 0.901 |

**False-positive rate on unseen-but-normal devices (Experiment B): 14.4%.**

Experiment A reproduces the paper's claimed 90–98% accuracy range. Experiment
B swaps the random packet-level split for a held-out-device split — entire
device files are set aside before training — and shows the accuracy that
looked solid in A doesn't hold up once the model is tested on traffic from
devices it has genuinely never seen. 14.4% of a brand-new, perfectly normal
device's traffic gets wrongly flagged as anomalous.

This matters because the authors' own evaluation (random packet-level
split) can't surface this: packets from the same device end up on both
sides of the split, so the model is partly being tested on devices it
already saw during training.
