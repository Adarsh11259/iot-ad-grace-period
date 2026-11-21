"""
Converts the raw .pcap packet captures from the IoT-AD authors' dataset
(Dataset/Profile trace/ in https://github.com/Hasniuj-Zahan/IoT-anomaly-detector)
into a single flat CSV of per-packet features.

The authors' own preprocess.py uses tshark and is hardcoded to one file and
one Windows path. This version uses scapy, walks every .pcap in the source
folder, and labels each row by which file (device) it came from.

Usage:
    python scripts/preprocess_pcaps.py /path/to/Profile\ trace data/profile_traces_combined.csv
"""
import os
import sys
import glob
import csv
from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP


def main(src_dir, out_csv):
    pcap_files = sorted(glob.glob(os.path.join(src_dir, "*.pcap")))
    print(f"Found {len(pcap_files)} pcap files in {src_dir}")

    rows = []
    for fpath in pcap_files:
        fname = os.path.basename(fpath)
        device_label = os.path.splitext(fname)[0]
        is_anomaly = 1 if device_label.lower() == "anomaly" else 0
        count = 0
        try:
            for pkt in PcapReader(fpath):
                if IP not in pkt:
                    continue
                length = len(pkt)
                src = pkt[IP].src
                dst = pkt[IP].dst
                if TCP in pkt:
                    dport = pkt[TCP].dport
                    flags = int(pkt[TCP].flags)
                elif UDP in pkt:
                    dport = pkt[UDP].dport
                    flags = -1
                else:
                    dport = -1
                    flags = -1
                rows.append([float(pkt.time), src, dst, length, dport, flags,
                             device_label, is_anomaly, fname])
                count += 1
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")
            continue
        print(f"{fname:30s} -> {count} IP packets")

    print(f"Total rows: {len(rows)}")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time", "SourceIP", "DestinationIP", "Length",
                     "DestinationPort", "FlagValue", "DeviceLabel",
                     "IsAnomaly", "SourceFile"])
        w.writerows(rows)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <profile_trace_dir> <output_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
