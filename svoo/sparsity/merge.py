#!/usr/bin/env python3
"""
Merge raw offline attention sparsity logs into a canonical CSV profile.

For each diffusion step, layer, and attention head, the merged profile keeps
the maximum sparsity observed across prompt samples.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def parse_sparsity_file(file_path):
    """
    Parse one raw sparsity log into per-step, per-layer, per-head values.

    Returns:
        dict: {(step, layer, head): sparsity_value}
    """
    data = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            current_step = None
            current_layer = None

            for line in f:
                # Example: [Sparsity] Layer X | Type: self | Step: Y
                header_match = re.match(r"\[Sparsity\] Layer (\d+) \| Type: \w+ \| Step: (\d+)", line)
                if header_match:
                    current_layer = int(header_match.group(1))
                    current_step = int(header_match.group(2))
                    continue

                # Example: Head  X: Sparsity=value
                head_match = re.match(r"\s*Head\s+(\d+):\s+Sparsity=([\d.]+)", line)
                if head_match and current_step is not None and current_layer is not None:
                    head_idx = int(head_match.group(1))
                    sparsity_value = float(head_match.group(2))
                    key = (current_step, current_layer, head_idx)
                    data[key] = max(data.get(key, 0.0), sparsity_value)

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

    return data


def merge_sparsity_data(file_paths):
    """
    Merge multiple raw logs using the maximum value for each profile entry.

    Returns:
        dict: {(step, layer, head): max_sparsity_value}
    """
    merged_data = defaultdict(list)

    for file_path in file_paths:
        print(f"Parsing {file_path}...")
        data = parse_sparsity_file(file_path)
        for key, value in data.items():
            merged_data[key].append(value)
        print(f"  Found {len(data)} entries")

    result = {}
    for key, values in merged_data.items():
        result[key] = max(values)

    return result


def save_to_csv(data, output_path):
    """
    Save merged sparsity values as Step, Layer, Head, Sparsity rows.

    Args:
        data: {(step, layer, head): sparsity_value}
        output_path: output CSV path
    """
    rows = []
    for (step, layer, head), sparsity in sorted(data.items()):
        rows.append(
            {
                "Step": step,
                "Layer": layer,
                "Head": head,
                "Sparsity": sparsity,
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Step", "Layer", "Head", "Sparsity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {output_path}")


def print_statistics(data):
    """Print summary statistics for the merged profile."""
    if not data:
        print("No data to display")
        return

    all_sparsities = list(data.values())
    avg_sparsity = sum(all_sparsities) / len(all_sparsities)

    step_stats = defaultdict(lambda: {"count": 0, "sum": 0.0, "values": []})
    for (step, layer, head), sparsity in data.items():
        step_stats[step]["count"] += 1
        step_stats[step]["sum"] += sparsity
        step_stats[step]["values"].append(sparsity)

    print("\n" + "=" * 80)
    print("SPARSITY STATISTICS")
    print("=" * 80)
    print(f"Total entries: {len(data)}")
    print(f"Overall average sparsity: {avg_sparsity:.6f}")
    print(f"Min sparsity: {min(all_sparsities):.6f}")
    print(f"Max sparsity: {max(all_sparsities):.6f}")

    print("\n" + "-" * 80)
    print("Statistics by Step:")
    print("-" * 80)
    print(f"{'Step':<10} {'Entries':<10} {'Avg Sparsity':<15} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    for step in sorted(step_stats.keys()):
        stats = step_stats[step]
        avg = stats["sum"] / stats["count"]
        min_val = min(stats["values"])
        max_val = max(stats["values"])
        print(f"{step:<10} {stats['count']:<10} {avg:<15.6f} {min_val:<10.6f} {max_val:<10.6f}")

    print("\n" + "-" * 80)
    print("Sample data (first 20 entries):")
    print("-" * 80)
    print(f"{'Step':<10} {'Layer':<10} {'Head':<10} {'Sparsity':<15}")
    print("-" * 80)
    count = 0
    for (step, layer, head), sparsity in sorted(data.items()):
        if count < 20:
            print(f"{step:<10} {layer:<10} {head:<10} {sparsity:<15.6f}")
            count += 1
        else:
            break

    if len(data) > 20:
        print(f"... (showing first 20 of {len(data)} entries)")

    print("=" * 80)


def validate_completeness(data, expected_steps=None, expected_layers=None, expected_heads=None):
    expected = {
        "steps": set(range(1, expected_steps + 1)) if expected_steps is not None else None,
        "layers": set(range(expected_layers)) if expected_layers is not None else None,
        "heads": set(range(expected_heads)) if expected_heads is not None else None,
    }
    if all(value is None for value in expected.values()):
        return

    observed_steps = {step for step, _, _ in data}
    observed_layers = {layer for _, layer, _ in data}
    observed_heads = {head for _, _, head in data}
    errors = []

    if expected["steps"] is not None:
        missing = sorted(expected["steps"] - observed_steps)
        if missing:
            errors.append(f"missing steps: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if expected["layers"] is not None:
        missing = sorted(expected["layers"] - observed_layers)
        if missing:
            errors.append(f"missing layers: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if expected["heads"] is not None:
        missing = sorted(expected["heads"] - observed_heads)
        if missing:
            errors.append(f"missing heads: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    if not errors and all(value is not None for value in expected.values()):
        expected_total = expected_steps * expected_layers * expected_heads
        if len(data) != expected_total:
            errors.append(f"expected {expected_total} entries, found {len(data)}")

    if errors:
        raise ValueError("Incomplete sparsity data: " + "; ".join(errors))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge attention_sparsity txt files into a Step,Layer,Head,Sparsity CSV.",
    )
    parser.add_argument("sparsity_dir", help="Directory containing attention_sparsity*.txt files.")
    parser.add_argument("output_csv", help="Output CSV path, e.g. sparsity_profiles/sparsity_wan_14B_t2v.csv.")
    parser.add_argument(
        "--pattern",
        default="attention_sparsity-*.txt",
        help="Glob pattern inside sparsity_dir. Default: attention_sparsity-*.txt",
    )
    parser.add_argument("--recursive", action="store_true", help="Search sparsity_dir recursively.")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Fail if fewer than this many sparsity files are found.",
    )
    parser.add_argument("--expected-steps", type=int, default=None, help="Fail if any 1-based step is missing.")
    parser.add_argument("--expected-layers", type=int, default=None, help="Fail if any zero-based layer is missing.")
    parser.add_argument("--expected-heads", type=int, default=None, help="Fail if any zero-based head is missing.")
    return parser.parse_args()


def find_input_files(sparsity_dir, pattern, recursive=False):
    target_dir = Path(sparsity_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory not found: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {target_dir}")

    glob_fn = target_dir.rglob if recursive else target_dir.glob
    return sorted(path for path in glob_fn(pattern) if path.is_file())


def main():
    args = parse_args()
    input_files = find_input_files(args.sparsity_dir, args.pattern, args.recursive)

    if args.expected_count is not None and len(input_files) < args.expected_count:
        raise FileNotFoundError(
            f"Expected at least {args.expected_count} sparsity files in {args.sparsity_dir}, "
            f"found {len(input_files)}"
        )
    if not input_files:
        raise FileNotFoundError(f"No sparsity files found in {args.sparsity_dir} with pattern {args.pattern}")

    print(f"Found {len(input_files)} sparsity files in {args.sparsity_dir}")

    print("Merging sparsity data from multiple files...")
    merged_data = merge_sparsity_data(input_files)
    
    if not merged_data:
        print("No data found in input files")
        return

    validate_completeness(
        merged_data,
        expected_steps=args.expected_steps,
        expected_layers=args.expected_layers,
        expected_heads=args.expected_heads,
    )

    save_to_csv(merged_data, args.output_csv)
    print_statistics(merged_data)


if __name__ == "__main__":
    main()
