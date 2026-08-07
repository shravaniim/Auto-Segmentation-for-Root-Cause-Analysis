from __future__ import annotations

import csv
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
DEV_FILE = BASE_DIR / "data" / "development_data_5000_shap.csv"
MON_FILE = BASE_DIR / "data" / "monitoring_data_5000_shap.csv"

NUMERIC_FEATURES = [
    "age",
    "income",
    "score",
    "target",
    "ead",
    "shap_age",
    "shap_income",
    "shap_region",
    "shap_occupation",
]
CATEGORICAL_FEATURES = ["region", "occupation"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MIN_SEGMENT_SIZE = 100


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def safe_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in {"na", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def describe_numeric(rows: List[Dict[str, str]], columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for col in columns:
        values = [safe_float(r[col]) for r in rows]
        values = [v for v in values if v is not None]
        if not values:
            continue
        summary[col] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def describe_categories(rows: List[Dict[str, str]], columns: Sequence[str]) -> Dict[str, Counter[str]]:
    return {col: Counter(r[col] for r in rows) for col in columns}


def print_summary(label: str, rows: List[Dict[str, str]]) -> None:
    print(f"=== {label} dataset ===")
    print(f"rows: {len(rows)}")
    numeric_summary = describe_numeric(rows, NUMERIC_FEATURES)
    for col, stats in numeric_summary.items():
        print(
            f"{col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"median={stats['median']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}"
        )
    for col, counts in describe_categories(rows, CATEGORICAL_FEATURES).items():
        print(f"{col}: {sorted(counts.items(), key=lambda item: -item[1])}")
    print()


def compare_datasets(dev: List[Dict[str, str]], mon: List[Dict[str, str]]) -> None:
    print("=== Dataset drift comparison ===")
    dev_numeric = describe_numeric(dev, NUMERIC_FEATURES)
    mon_numeric = describe_numeric(mon, NUMERIC_FEATURES)
    for col in NUMERIC_FEATURES:
        if col in dev_numeric and col in mon_numeric:
            dev_mean = dev_numeric[col]["mean"]
            mon_mean = mon_numeric[col]["mean"]
            diff = mon_mean - dev_mean
            print(f"{col}: dev_mean={dev_mean:.4f}, mon_mean={mon_mean:.4f}, diff={diff:.4f}")
    for col in CATEGORICAL_FEATURES:
        dev_counts = describe_categories(dev, [col])[col]
        mon_counts = describe_categories(mon, [col])[col]
        print(f"{col} distribution:")
        for key in sorted(set(dev_counts) | set(mon_counts)):
            print(f"  {key}: dev={dev_counts.get(key,0)}, mon={mon_counts.get(key,0)}")
    print()


def subset_rows(rows: Iterable[Dict[str, str]], conditions: Dict[str, str]) -> List[Dict[str, str]]:
    return [row for row in rows if all(row.get(col) == value for col, value in conditions.items())]


def numeric_split_candidates(rows: List[Dict[str, str]], col: str, quantiles: Sequence[float]) -> List[Tuple[str, float]]:
    values = sorted(v for v in (safe_float(r[col]) for r in rows) if v is not None)
    candidates = []
    if not values:
        return candidates
    for q in quantiles:
        index = int(q * (len(values) - 1))
        threshold = values[index]
        candidates.append((f"{col} <= {threshold:.4f}", threshold))
    return candidates


def drift_score(dev: List[Dict[str, str]], mon: List[Dict[str, str]]) -> float:
    dev_size = len(dev)
    mon_size = len(mon)
    if dev_size == 0 or mon_size == 0:
        return 0.0

    score = 0.0
    for col in NUMERIC_FEATURES:
        dev_values = [safe_float(r[col]) for r in dev]
        mon_values = [safe_float(r[col]) for r in mon]
        dev_values = [v for v in dev_values if v is not None]
        mon_values = [v for v in mon_values if v is not None]
        if not dev_values or not mon_values:
            continue
        dev_mean = statistics.mean(dev_values)
        mon_mean = statistics.mean(mon_values)
        denom = abs(dev_mean) + abs(mon_mean) + 1e-9
        score += abs(mon_mean - dev_mean) / denom

    for col in CATEGORICAL_FEATURES:
        dev_counts = Counter(row[col] for row in dev)
        mon_counts = Counter(row[col] for row in mon)
        keys = set(dev_counts) | set(mon_counts)
        for key in keys:
            dev_ratio = dev_counts.get(key, 0) / dev_size
            mon_ratio = mon_counts.get(key, 0) / mon_size
            score += abs(mon_ratio - dev_ratio)

    return score


def find_drift_segments(dev: List[Dict[str, str]], mon: List[Dict[str, str]], top_n: int = 6) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []

    for cat in CATEGORICAL_FEATURES:
        values = sorted(set(row[cat] for row in dev + mon))
        for val in values:
            dev_sub = subset_rows(dev, {cat: val})
            mon_sub = subset_rows(mon, {cat: val})
            if len(dev_sub) < MIN_SEGMENT_SIZE or len(mon_sub) < MIN_SEGMENT_SIZE:
                continue
            candidates.append({
                "segment": f"{cat}={val}",
                "size_dev": len(dev_sub),
                "size_mon": len(mon_sub),
                "score": drift_score(dev_sub, mon_sub),
            })

    for col in ["age", "income", "ead", "score"]:
        for label, threshold in numeric_split_candidates(dev + mon, col, [0.25, 0.5, 0.75]):
            dev_sub = [row for row in dev if safe_float(row[col]) is not None and safe_float(row[col]) <= threshold]
            mon_sub = [row for row in mon if safe_float(row[col]) is not None and safe_float(row[col]) <= threshold]
            if len(dev_sub) >= MIN_SEGMENT_SIZE and len(mon_sub) >= MIN_SEGMENT_SIZE:
                candidates.append({
                    "segment": label,
                    "size_dev": len(dev_sub),
                    "size_mon": len(mon_sub),
                    "score": drift_score(dev_sub, mon_sub),
                })

            dev_sub = [row for row in dev if safe_float(row[col]) is not None and safe_float(row[col]) > threshold]
            mon_sub = [row for row in mon if safe_float(row[col]) is not None and safe_float(row[col]) > threshold]
            if len(dev_sub) >= MIN_SEGMENT_SIZE and len(mon_sub) >= MIN_SEGMENT_SIZE:
                candidates.append({
                    "segment": label.replace("<=", ">"),
                    "size_dev": len(dev_sub),
                    "size_mon": len(mon_sub),
                    "score": drift_score(dev_sub, mon_sub),
                })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_n]


def one_hot_encode(rows: List[Dict[str, str]], cols: Sequence[str]) -> Tuple[List[List[float]], Dict[str, Dict[str, int]]]:
    mappings: Dict[str, Dict[str, int]] = {}
    for col in cols:
        values = sorted(set(row[col] for row in rows))
        mappings[col] = {value: idx for idx, value in enumerate(values)}

    vectors: List[List[float]] = []
    for row in rows:
        vector: List[float] = []
        for col in cols:
            encoding = [0.0] * len(mappings[col])
            value = row[col]
            index = mappings[col].get(value)
            if index is not None:
                encoding[index] = 1.0
            vector.extend(encoding)
        vectors.append(vector)
    return vectors, mappings


def build_feature_vectors(rows: List[Dict[str, str]]) -> List[List[float]]:
    base_numeric = []
    for row in rows:
        values = []
        for col in ["age", "income", "ead", "score", "shap_age", "shap_income", "shap_region", "shap_occupation"]:
            values.append(safe_float(row[col]) or 0.0)
        base_numeric.append(values)

    cat_vectors, mappings = one_hot_encode(rows, CATEGORICAL_FEATURES)
    vectors: List[List[float]] = []
    for numeric, cat in zip(base_numeric, cat_vectors):
        vectors.append(numeric + cat)
    return vectors


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def kmeans(data: List[List[float]], k: int = 4, max_iter: int = 100, seed: int = 42) -> Tuple[List[List[float]], List[int]]:
    random.seed(seed)
    centroids = [list(point) for point in random.sample(data, k)]
    assignments = [0] * len(data)

    for iteration in range(max_iter):
        changed = False
        for i, point in enumerate(data):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            best = min(range(k), key=lambda idx: distances[idx])
            if assignments[i] != best:
                changed = True
            assignments[i] = best

        if not changed:
            break

        clusters: List[List[List[float]]] = [[] for _ in range(k)]
        for assignment, point in zip(assignments, data):
            clusters[assignment].append(point)

        for idx in range(k):
            if clusters[idx]:
                centroid = [statistics.mean(col) for col in zip(*clusters[idx])]
                centroids[idx] = centroid

    return centroids, assignments


def summarize_clusters(dev: List[Dict[str, str]], mon: List[Dict[str, str]], assignments: List[int], labels: List[str], k: int) -> List[Dict[str, object]]:
    combined = dev + mon
    dev_size = len(dev)
    mon_size = len(mon)
    cluster_summary: List[Dict[str, object]] = []

    for cluster_id in range(k):
        dev_count = sum(1 for idx, row in enumerate(combined[:dev_size]) if assignments[idx] == cluster_id)
        mon_count = sum(1 for idx, row in enumerate(combined[dev_size:], start=dev_size) if assignments[idx] == cluster_id)
        segment_rows = [row for idx, row in enumerate(combined) if assignments[idx] == cluster_id]
        numeric_stats = describe_numeric(segment_rows, ["age", "income", "score", "ead"])
        cluster_summary.append({
            "cluster_id": cluster_id,
            "dev_count": dev_count,
            "mon_count": mon_count,
            "share_dev": dev_count / dev_size if dev_size else 0.0,
            "share_mon": mon_count / mon_size if mon_size else 0.0,
            "numeric_stats": numeric_stats,
        })
    cluster_summary.sort(key=lambda item: abs(item["share_mon"] - item["share_dev"]), reverse=True)
    return cluster_summary


def print_cluster_report(cluster_summary: List[Dict[str, object]]) -> None:
    print("=== K-means cluster drift report ===")
    for cluster in cluster_summary:
        print(
            f"cluster={cluster['cluster_id']} dev={cluster['dev_count']} mon={cluster['mon_count']} "
            f"share_dev={cluster['share_dev']:.3f} share_mon={cluster['share_mon']:.3f}"
        )
        stats = cluster["numeric_stats"]
        for col, values in stats.items():
            print(f"  {col}: mean={values['mean']:.2f}, median={values['median']:.2f}, min={values['min']:.2f}, max={values['max']:.2f}")
        print()


def main() -> None:
    dev_rows = read_csv_rows(DEV_FILE)
    mon_rows = read_csv_rows(MON_FILE)

    print_summary("development", dev_rows)
    print_summary("monitoring", mon_rows)
    compare_datasets(dev_rows, mon_rows)

    print("=== Drift localization tree candidates ===")
    top_segments = find_drift_segments(dev_rows, mon_rows, top_n=8)
    for segment in top_segments:
        print(
            f"segment={segment['segment']} score={segment['score']:.4f} "
            f"dev={segment['size_dev']} mon={segment['size_mon']}"
        )
    print()

    print("=== K-means segment discovery ===")
    combined_rows = dev_rows + mon_rows
    feature_vectors = build_feature_vectors(combined_rows)
    centroids, assignments = kmeans(feature_vectors, k=5)
    cluster_summary = summarize_clusters(dev_rows, mon_rows, assignments, ["dev", "mon"], k=5)
    print_cluster_report(cluster_summary)


if __name__ == "__main__":
    main()
