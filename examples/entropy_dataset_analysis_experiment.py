import argparse
import csv
import datetime
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from examples.batch_static_skip_experiment import (
    aggregate_batch_summary,
    build_task_specs,
    case_summary,
    parse_ids,
    read_json,
    run_task_got,
    selected_tasks,
)
from examples.static_skip_experiment import (
    reduction_ratio,
    select_low_entropy_thoughts,
    token_summary,
    write_candidate_ranking,
)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
            "std": None,
            "range": None,
            "iqr": None,
            "coefficient_of_variation": None,
        }
    avg = mean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    p25 = percentile(values, 0.25)
    p75 = percentile(values, 0.75)
    return {
        "count": len(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p25": p25,
        "median": percentile(values, 0.50),
        "p75": p75,
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": avg,
        "std": std,
        "range": max(values) - min(values),
        "iqr": (p75 - p25) if p25 is not None and p75 is not None else None,
        "coefficient_of_variation": (std / avg) if avg else None,
    }


def entropy_separation(candidates: List[Dict[str, Any]], skip_ratio: float) -> Dict[str, Any]:
    entropies = [float(item["entropy"]) for item in candidates]
    count = int(len(entropies) * skip_ratio)
    if count <= 0:
        return {
            "low_count": 0,
            "high_count": 0,
            "low_mean": None,
            "high_mean": None,
            "mean_gap": None,
            "mean_gap_over_std": None,
            "overlap_low_max_high_min": None,
        }
    ordered = sorted(entropies)
    low = ordered[:count]
    high = ordered[-count:]
    std = pstdev(entropies) if len(entropies) > 1 else 0.0
    gap = mean(high) - mean(low)
    return {
        "low_count": len(low),
        "high_count": len(high),
        "low_mean": mean(low),
        "high_mean": mean(high),
        "mean_gap": gap,
        "mean_gap_over_std": (gap / std) if std else None,
        "overlap_low_max_high_min": max(low) >= min(high),
    }


def response_features(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    stripped = text.strip()
    lower = stripped.lower()
    tokens = metadata.get("tokens") or []
    return {
        "response_chars": len(stripped),
        "observed_tokens": metadata.get("num_observed_tokens"),
        "metadata_token_count": len(tokens),
        "starts_with_list": stripped.startswith("["),
        "starts_with_json": stripped.startswith("{"),
        "has_markdown_fence": "```" in stripped,
        "has_xml_tag": "<" in stripped and ">" in stripped,
        "contains_explanation_marker": any(
            marker in lower
            for marker in [
                "therefore",
                "because",
                "reason",
                "step",
                "first",
                "finally",
                "no changes are needed",
            ]
        ),
        "line_count": stripped.count("\n") + (1 if stripped else 0),
    }


def collect_entropy_candidates(
    full_graph_json: List[Dict[str, Any]],
    entropy_field: str,
) -> List[Dict[str, Any]]:
    _, _, candidates = select_low_entropy_thoughts(
        full_graph_json,
        0.0,
        entropy_field,
        "low",
    )
    operation_records = {
        record.get("operation_index"): record
        for record in full_graph_json
        if isinstance(record, dict) and "operation" in record
    }
    enriched = []
    for candidate in candidates:
        record = operation_records.get(candidate["operation_index"], {})
        thought_index = candidate.get("thought_index")
        metadata = {}
        thought_state = {}
        if isinstance(thought_index, int):
            metadata_list = record.get("thought_metadata", [])
            thought_list = record.get("thoughts", [])
            if thought_index < len(metadata_list):
                metadata = metadata_list[thought_index] or {}
            if thought_index < len(thought_list):
                thought_state = thought_list[thought_index] or {}
        response_text = str(metadata.get("response_text", ""))
        enriched_item = dict(candidate)
        enriched_item.update(
            {
                "response_preview": response_text[:500],
                "prompt_preview": str(metadata.get("prompt", ""))[:500],
                "current_preview": str(thought_state.get("current", ""))[:500],
                "sum_entropy_bits": metadata.get("sum_entropy_bits"),
                "avg_entropy_bits": metadata.get("avg_entropy_bits"),
                "normalized_sum_entropy_bits": metadata.get(
                    "normalized_sum_entropy_bits"
                ),
                "normalized_avg_entropy_bits": metadata.get(
                    "normalized_avg_entropy_bits"
                ),
                **response_features(response_text, metadata),
            }
        )
        enriched.append(enriched_item)
    return enriched


def select_random_candidates(
    candidates: List[Dict[str, Any]],
    skip_ratio: float,
    rng: random.Random,
) -> Tuple[Dict[int, set], Dict[int, set], List[Dict[str, Any]]]:
    target = int(len(candidates) * skip_ratio)
    shuffled = [dict(item) for item in candidates]
    rng.shuffle(shuffled)

    thought_counts: Dict[int, int] = Counter()
    refine_counts: Dict[int, int] = Counter()
    for item in shuffled:
        if item["candidate_type"] == "validate_and_improve_refine":
            refine_counts[item["operation_index"]] += 1
        else:
            thought_counts[item["operation_index"]] += 1

    selected = []
    selected_thought_counts: Dict[int, int] = Counter()
    selected_refine_counts: Dict[int, int] = Counter()
    for item in shuffled:
        if len(selected) >= target:
            break
        operation_index = item["operation_index"]
        if item["candidate_type"] == "validate_and_improve_refine":
            if selected_refine_counts[operation_index] >= refine_counts[operation_index] - 1:
                item["random_skip_exclusion_reason"] = (
                    "keep_at_least_one_refine_candidate_per_operation"
                )
                continue
            selected_refine_counts[operation_index] += 1
        else:
            if selected_thought_counts[operation_index] >= thought_counts[operation_index] - 1:
                item["random_skip_exclusion_reason"] = (
                    "keep_at_least_one_thought_candidate_per_operation"
                )
                continue
            selected_thought_counts[operation_index] += 1
        item["selected_for_random_skip"] = True
        selected.append(item)

    skip_thought_indices: Dict[int, set] = defaultdict(set)
    skip_refine_indices: Dict[int, set] = defaultdict(set)
    for item in selected:
        if item["candidate_type"] == "validate_and_improve_refine":
            skip_refine_indices[item["operation_index"]].add(
                (item["input_thought_index"], item["try_index"])
            )
        else:
            skip_thought_indices[item["operation_index"]].add(item["thought_index"])
    return dict(skip_thought_indices), dict(skip_refine_indices), selected


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, set))
                    else value
                    for key, value in row.items()
                }
            )


def average_metric(rows: List[Dict[str, Any]], path: Sequence[str]) -> Optional[float]:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key, None) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return mean(values) if values else None


def summarize_random_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_trials": len(trials),
        "avg_total_token_reduction": average_metric(
            trials, ["reductions", "total_token_reduction"]
        ),
        "avg_api_call_reduction": average_metric(
            trials, ["reductions", "api_call_reduction"]
        ),
        "avg_cost_reduction": average_metric(trials, ["reductions", "cost_reduction"]),
        "avg_latency_reduction": average_metric(
            trials, ["reductions", "latency_reduction"]
        ),
        "avg_random_selected_entropy": average_metric(
            trials, ["random_baseline", "selected_entropy_mean"]
        ),
        "full_accuracy": average_metric(trials, ["full", "solved"]),
        "compressed_accuracy": average_metric(trials, ["compressed", "solved"]),
    }


def full_graph_path(input_run_root: Optional[str], task_name: str, case_id: int) -> Optional[str]:
    if not input_run_root:
        return None
    path = os.path.join(input_run_root, task_name, f"id{case_id}", "full_graph.json")
    return path if os.path.exists(path) else None


def run_case(
    task: Any,
    case: Dict[str, Any],
    args: argparse.Namespace,
    run_root: str,
) -> Dict[str, Any]:
    case_dir = os.path.join(run_root, task.name, f"id{case['id']}")
    os.makedirs(case_dir, exist_ok=True)

    reused_full_path = full_graph_path(args.input_run_root, task.name, case["id"])
    if reused_full_path:
        full_path = reused_full_path
        full_json = read_json(full_path)
    else:
        full_path = os.path.join(case_dir, "full_graph.json")
        _, full_json = run_task_got(
            task,
            args.config_path,
            args.model_name,
            case,
            full_path,
        )

    candidates = collect_entropy_candidates(full_json, args.entropy_field)
    ranking_json_path = os.path.join(case_dir, "entropy_candidates.json")
    ranking_csv_path = os.path.join(case_dir, "entropy_candidates.csv")
    write_candidate_ranking(candidates, ranking_json_path, ranking_csv_path)

    entropies = [float(item["entropy"]) for item in candidates]
    low_cutoff_count = int(len(candidates) * args.skip_ratio)
    low_entropy_nodes = sorted(candidates, key=lambda item: item["entropy"])[
        :low_cutoff_count
    ]
    high_entropy_nodes = sorted(candidates, key=lambda item: item["entropy"], reverse=True)[
        :low_cutoff_count
    ]
    write_csv(low_entropy_nodes, os.path.join(case_dir, "low_entropy_nodes.csv"))
    write_csv(high_entropy_nodes, os.path.join(case_dir, "high_entropy_nodes.csv"))

    random_trial_summaries = []
    rng = random.Random(args.random_seed + int(case["id"]))
    for trial_index in range(args.random_trials):
        trial_seed = rng.randrange(0, 2**31)
        trial_rng = random.Random(trial_seed)
        skip_thought_indices, skip_refine_indices, selected = select_random_candidates(
            candidates,
            args.skip_ratio,
            trial_rng,
        )
        trial_dir = os.path.join(case_dir, "random", f"trial{trial_index:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        compressed_path = os.path.join(trial_dir, "compressed_graph.json")
        _, compressed_json = run_task_got(
            task,
            args.config_path,
            args.model_name,
            case,
            compressed_path,
            skip_thought_indices=skip_thought_indices,
            skip_refine_indices=skip_refine_indices,
        )
        selected_entropies = [float(item["entropy"]) for item in selected]
        trial_summary = case_summary(
            task,
            case,
            args.model_name,
            args.entropy_field,
            args.skip_ratio,
            "random",
            full_json,
            compressed_json,
            candidates,
            skip_thought_indices,
            skip_refine_indices,
            {
                "full_graph": full_path,
                "compressed_graph": compressed_path,
                "summary": os.path.join(trial_dir, "summary.json"),
            },
        )
        trial_summary["random_baseline"] = {
            "trial_index": trial_index,
            "random_seed": trial_seed,
            "selected_entropy_mean": mean(selected_entropies)
            if selected_entropies
            else None,
            "selected_entropy_distribution": describe(selected_entropies),
            "selected_nodes": [
                {
                    "node_label": item.get("node_label"),
                    "operation": item.get("operation"),
                    "entropy": item.get("entropy"),
                }
                for item in selected
            ],
        }
        with open(trial_summary["paths"]["summary"], "w", encoding="utf-8") as f:
            json.dump(trial_summary, f, indent=2, ensure_ascii=False)
        random_trial_summaries.append(trial_summary)

    full_tokens = token_summary(full_json)
    case_result = {
        "task": task.name,
        "method": task.method_name,
        "case_id": case["id"],
        "model_name": args.model_name,
        "entropy_field": args.entropy_field,
        "skip_ratio": args.skip_ratio,
        "num_candidates": len(candidates),
        "full": full_tokens,
        "entropy_distribution": describe(entropies),
        "entropy_separation": entropy_separation(candidates, args.skip_ratio),
        "operation_type_counts": dict(Counter(item["operation"] for item in candidates)),
        "low_entropy_operation_type_counts": dict(
            Counter(item["operation"] for item in low_entropy_nodes)
        ),
        "high_entropy_operation_type_counts": dict(
            Counter(item["operation"] for item in high_entropy_nodes)
        ),
        "low_entropy_nodes_path": os.path.join(case_dir, "low_entropy_nodes.csv"),
        "high_entropy_nodes_path": os.path.join(case_dir, "high_entropy_nodes.csv"),
        "entropy_candidates_json": ranking_json_path,
        "entropy_candidates_csv": ranking_csv_path,
        "random_baseline": summarize_random_trials(random_trial_summaries),
        "random_trials": random_trial_summaries,
    }
    with open(os.path.join(case_dir, "entropy_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(case_result, f, indent=2, ensure_ascii=False)
    return case_result


def aggregate_case_results(case_results: List[Dict[str, Any]], run_root: str) -> Dict[str, Any]:
    random_runs = [
        trial
        for case_result in case_results
        for trial in case_result.get("random_trials", [])
    ]
    return {
        "num_cases": len(case_results),
        "tasks": sorted({item["task"] for item in case_results}),
        "avg_num_candidates": average_metric(case_results, ["num_candidates"]),
        "avg_entropy_mean": average_metric(
            case_results, ["entropy_distribution", "mean"]
        ),
        "avg_entropy_std": average_metric(case_results, ["entropy_distribution", "std"]),
        "avg_entropy_range": average_metric(
            case_results, ["entropy_distribution", "range"]
        ),
        "avg_entropy_iqr": average_metric(case_results, ["entropy_distribution", "iqr"]),
        "avg_low_high_mean_gap": average_metric(
            case_results, ["entropy_separation", "mean_gap"]
        ),
        "random_baseline": summarize_random_trials(random_runs),
        "random_batch_summary": aggregate_batch_summary(
            random_runs,
            {
                "run_root": run_root,
                "analysis_summary": os.path.join(run_root, "entropy_analysis_summary.json"),
            },
        ),
        "cases": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze entropy distribution and random-skip baseline over a dataset."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--config-path",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "graph_of_thoughts",
            "language_models",
            "config.json",
        ),
    )
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--data-ids", default=None)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--skip-ratio", type=float, default=0.2)
    parser.add_argument("--random-trials", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--entropy-field", default="normalized_avg_entropy_bits")
    parser.add_argument(
        "--input-run-root",
        default=None,
        help="Reuse existing full_graph.json files from this batch run root.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "entropy_analysis_results"),
    )
    args = parser.parse_args()

    if not 0.0 <= args.skip_ratio <= 1.0:
        raise ValueError("--skip-ratio must be between 0 and 1")
    if args.random_trials < 0:
        raise ValueError("--random-trials must be non-negative")

    task_specs = build_task_specs()
    tasks = selected_tasks(task_specs, args.tasks)
    data_ids = parse_ids(args.data_ids)
    max_cases = None if data_ids is not None else args.max_cases
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = os.path.join(
        args.output_dir,
        f"{args.model_name}_random{args.random_trials}_skip{args.skip_ratio}_{timestamp}",
    )
    os.makedirs(run_root, exist_ok=True)

    case_results = []
    for task in tasks:
        cases = task.load_cases(data_ids, max_cases)
        for case in cases:
            print(f"Running entropy_analysis {task.name} id={case['id']}")
            case_results.append(run_case(task, case, args, run_root))

    summary = aggregate_case_results(case_results, run_root)
    summary_path = os.path.join(run_root, "entropy_analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
