# scripts/split_qrels.py
import argparse
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="z.B. data/processed/jobs_interamt_expert")
parser.add_argument("--test-frac", type=float, default=0.2, help="Anteil der Queries fürs Test-Set")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

DATASET_DIR = Path(args.dataset)
QRELS_PATH = DATASET_DIR / "qrels" / "test.tsv"
QRELS_ALL_BACKUP = DATASET_DIR / "qrels" / "all.tsv"

random.seed(args.seed)

# Alle Qrels einlesen
rows = []
with open(QRELS_PATH, "r", encoding="utf-8") as f:
    header = next(f)
    for line in f:
        qid, docid, score = line.strip().split("\t")
        rows.append((qid, docid, score))

qids = sorted({qid for qid, _, _ in rows})
random.shuffle(qids)

n_test = max(1, round(len(qids) * args.test_frac))
test_qids = set(qids[:n_test])
train_qids = set(qids[n_test:])

print(f"Queries gesamt: {len(qids)}  ->  Train: {len(train_qids)}  Test: {len(test_qids)}")

# Backup des ungesplitteten Files, falls noch nicht vorhanden
if not QRELS_ALL_BACKUP.exists():
    QRELS_ALL_BACKUP.write_text(header + "".join(f"{q}\t{d}\t{s}\n" for q, d, s in rows), encoding="utf-8")
    print(f"Backup der vollständigen Qrels: {QRELS_ALL_BACKUP}")

train_path = DATASET_DIR / "qrels" / "train.tsv"
test_path = DATASET_DIR / "qrels" / "test.tsv"

with open(train_path, "w", encoding="utf-8") as f:
    f.write(header)
    for q, d, s in rows:
        if q in train_qids:
            f.write(f"{q}\t{d}\t{s}\n")

with open(test_path, "w", encoding="utf-8") as f:
    f.write(header)
    for q, d, s in rows:
        if q in test_qids:
            f.write(f"{q}\t{d}\t{s}\n")

print(f"Geschrieben: {train_path} und {test_path}")