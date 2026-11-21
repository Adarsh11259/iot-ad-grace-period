# Extending IoT-AD: a generalization gap, and a grace period that fixes most of it

Course project building on:

> Zahan, H., Al Azad, M. W., Ali, I., & Mastorakis, S. (2024). *IoT-AD: A
> Framework to Detect Anomalies Among Interconnected IoT Devices.* IEEE
> Internet of Things Journal.

using the authors' own code and dataset:
[Hasniuj-Zahan/IoT-anomaly-detector](https://github.com/Hasniuj-Zahan/IoT-anomaly-detector).

## What IoT-AD does

Packet-level anomaly detection (a supervised classifier, Random Forest in
the authors' own scripts) plus a rule-based device-interaction tree that
can trace a fault back to its root cause and roll it back.

## What we found

The authors evaluate their packet-level detector with a random 75/25
split over pooled packets. That lets packets from the same device land in
both train and test. We reproduced their method (96.6% accuracy, matching
their claimed range), then re-ran the exact same model on a held-out-device
split — entire device files set aside before training — to test whether it
actually generalizes to devices it has never seen.

**Result:** accuracy drops to 90.8%, and 14.4% of a brand-new, perfectly
normal device's traffic gets wrongly flagged as an anomaly. See
[`results/gap_analysis_results.md`](results/gap_analysis_results.md).

## The new idea: a grace period for new devices

If a new device gets flagged this often just for being new, give it a
grace period: let it build its own short behavioral baseline before
judging it against the network at large, instead of throwing it straight
at the shared model.

We implemented this as a per-device Isolation Forest trained only on each
held-out device's first 30% of traffic, then tested the remaining 70%
against that private baseline instead of the shared Random Forest model.

**Result:** overall false-positive rate on unseen devices drops from
10.8% to 2.2%. See
[`results/calibration_results.md`](results/calibration_results.md).

**Known limitation, documented but not fixed:** a device compromised
before onboarding could poison its own baseline during the grace period.
Network-isolating new devices during calibration is a reasonable fix we
propose but didn't build.

## Repo layout

```
data/        derived packet-feature CSV + how to regenerate it
scripts/     preprocess_pcaps.py, gap_analysis.py, calibration_demo.py
results/     the two result tables above
exploration/ things we tried and explicitly dropped, and why
```

## Reproducing this

```bash
pip install -r requirements.txt
python scripts/gap_analysis.py
python scripts/calibration_demo.py
```

## Credit

Built on top of the IoT-AD paper and the authors' own public repository
(linked above). This project only reuses their dataset and the Random
Forest approach that's directly verifiable in their code; everything in
`scripts/` and `results/` is our own analysis and implementation.
