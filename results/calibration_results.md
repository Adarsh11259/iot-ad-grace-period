# Calibration / grace-period results

`scripts/calibration_demo.py`, run on the same 6 held-out devices as the gap
analysis. "Naive" = judged directly by the shared global Random Forest
model. "Calibrated" = judged by a private Isolation Forest model built only
from that device's own first 30% of traffic (the grace period).

| Device | Calibration packets | Test packets | Naive FP% | Calibrated FP% |
|---|---|---|---|---|
| amazonplug.pcap | 1,591 | 3,715 | 23.4% | 3.3% |
| echo.pcap | 2,344 | 5,471 | 0.7% | 1.4% |
| ecobee.pcap | 18 | 43 | 14.0% | 9.3% |
| on and off.pcap | 24 | 56 | 7.1% | 8.9% |
| onkettle.pcap | 637 | 1,489 | 16.0% | 1.6% |
| switchonteckin.pcap | 21 | 49 | 20.4% | 8.2% |
| **Overall** | — | 10,823 | **10.8%** | **2.2%** |

Giving a new device a short grace period — learning its own normal pattern
before judging it against the network at large — cuts the overall false
positive rate on unseen devices from 10.8% to 2.2%.

**Known limitation (documented, not fixed):** the calibration window
trusts a device's behavior while learning it. A device that's already
compromised at the moment it's onboarded could poison its own baseline
during the grace period, teaching the system that its malicious behavior
is "normal." A reasonable mitigation — network-isolating/quarantining a
new device during its calibration window so it can't do real harm even if
compromised — is proposed here but not implemented.
