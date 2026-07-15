import argparse
import csv
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_of_thoughts import controller
from graph_of_thoughts.language_models.factory import build_language_model

from examples.sorting import sorting_032


SKIPPABLE_OPERATION_TYPES = {
    "generate",
    "aggregate",
    "improve",
    "validate_and_improve",
}


def load_sorting_case(data_id: int) -> Dict[str, Any]:
    data_path = os.path.join(
        os.path.dirname(__file__), "sorting", "sorting_032.csv"
    )
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[0]) == data_id:
                return {
                    "id": int(row[0]),
                    "original": row[1],
                    "ground_truth": row[2],
                }
    raise ValueError(f"Could not find sorting_032 data id {data_id}")


def run_sorting_got(
    config_path: str,
    model_name: str,
    problem: Dict[str, Any],
    output_path: str,
    skip_operation_indices: set = None,
) -> Tuple[controller.Controller, Dict[str, Any]]:
    lm = build_language_model(config_path, model_name, cache=False)
    graph = sorting_032.got()
    ctrl = controller.Controller(
        lm,
        graph,
        sorting_032.SortingPrompter(),
        sorting_032.SortingParser(),
        {
            "original": problem["original"],
            "current": "",
            "phase": 0,
            "method": "got",
        },
        skip_operation_indices=skip_operation_indices or set(),
    )
    ctrl.run()
    ctrl.output_graph(output_path)
    return ctrl, read_json(output_path)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def operation_entropy(
    operation_record: Dict[str, Any],
    entropy_field: str,
) -> float:
    values = []
    for metadata in operation_record.get("thought_metadata", []):
        value = metadata.get(entropy_field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return float("inf")
    return sum(values) / len(values)


def select_low_entropy_operations(
    full_graph_json: List[Dict[str, Any]],
    skip_ratio: float,
    entropy_field: str,
) -> Tuple[set, List[Dict[str, Any]]]:
    candidates = []
    for record in full_graph_json:
        if "operation" not in record:
            continue
        if record["operation"] not in SKIPPABLE_OPERATION_TYPES:
            continue
        # root generate 通常会创建初始计划或拆分结果。
        # 跳过它会破坏诸如 "part" 这样的任务特定字段，因此第一个静态实验保留它。
        if not record.get("predecessors"):
            continue
        entropy = operation_entropy(record, entropy_field)
        if entropy == float("inf"):
            continue
        candidates.append(
            {
                "operation_index": record["operation_index"],
                "operation_id": record["operation_id"],
                "operation": record["operation"],
                "entropy": entropy,
            }
        )

    candidates.sort(key=lambda item: item["entropy"])
    num_to_skip = int(len(candidates) * skip_ratio)
    selected = candidates[:num_to_skip]
    return {item["operation_index"] for item in selected}, candidates


def final_solved(graph_json: List[Dict[str, Any]]) -> bool:
    for record in reversed(graph_json):
        solved = record.get("problem_solved")
        if isinstance(solved, list) and solved:
            return any(bool(item) for item in solved)
    return False


def token_summary(graph_json: List[Dict[str, Any]]) -> Dict[str, float]:
    totals = graph_json[-1]
    prompt_tokens = float(totals.get("prompt_tokens", 0) or 0)
    completion_tokens = float(totals.get("completion_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": float(
            totals.get("total_tokens", prompt_tokens + completion_tokens) or 0
        ),
        "api_calls": float(totals.get("api_calls", 0) or 0),
        "total_latency_seconds": float(
            totals.get("total_latency_seconds", 0.0) or 0.0
        ),
        "cost": float(totals.get("cost", 0.0) or 0.0),
    }


def reduction_ratio(full_value: float, compressed_value: float) -> float:
    if full_value <= 0:
        return 0.0
    return (full_value - compressed_value) / full_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a static low-entropy [SKIP] replay experiment on sorting_032 GoT."
    )
    parser.add_argument("--data-id", type=int, default=0)
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
    parser.add_argument("--skip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-field", default="avg_entropy_bits")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "static_skip_results"),
    )
    args = parser.parse_args()

    if not 0.0 <= args.skip_ratio <= 1.0:
        raise ValueError("--skip-ratio must be between 0 and 1")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(
        args.output_dir,
        f"sorting032_{args.model_name}_id{args.data_id}_skip{args.skip_ratio}_{timestamp}",
    )
    os.makedirs(run_dir, exist_ok=True)

    problem = load_sorting_case(args.data_id)
    full_path = os.path.join(run_dir, "full_graph.json")
    compressed_path = os.path.join(run_dir, "compressed_graph.json")
    summary_path = os.path.join(run_dir, "summary.json")

    _, full_json = run_sorting_got(
        args.config_path,
        args.model_name,
        problem,
        full_path,
    )
    skip_indices, ranked_candidates = select_low_entropy_operations(
        full_json,
        args.skip_ratio,
        args.entropy_field,
    )
    _, compressed_json = run_sorting_got(
        args.config_path,
        args.model_name,
        problem,
        compressed_path,
        skip_operation_indices=skip_indices,
    )

    full_tokens = token_summary(full_json)
    compressed_tokens = token_summary(compressed_json)
    summary = {
        "task": "sorting_032_got",
        "data_id": args.data_id,
        "model_name": args.model_name,
        "entropy_field": args.entropy_field,
        "skip_ratio": args.skip_ratio,
        "selected_skip_operation_indices": sorted(skip_indices),
        "num_candidates": len(ranked_candidates),
        "num_skipped_operations": len(skip_indices),
        "full": {
            "solved": final_solved(full_json),
            **full_tokens,
        },
        "compressed": {
            "solved": final_solved(compressed_json),
            **compressed_tokens,
        },
        "reductions": {
            "prompt_token_reduction": reduction_ratio(
                full_tokens["prompt_tokens"], compressed_tokens["prompt_tokens"]
            ),
            "completion_token_reduction": reduction_ratio(
                full_tokens["completion_tokens"],
                compressed_tokens["completion_tokens"],
            ),
            "total_token_reduction": reduction_ratio(
                full_tokens["total_tokens"], compressed_tokens["total_tokens"]
            ),
            "cost_reduction": reduction_ratio(
                full_tokens["cost"], compressed_tokens["cost"]
            ),
            "api_call_reduction": reduction_ratio(
                full_tokens["api_calls"], compressed_tokens["api_calls"]
            ),
            "latency_reduction": reduction_ratio(
                full_tokens["total_latency_seconds"],
                compressed_tokens["total_latency_seconds"],
            ),
            "operation_skip_ratio": (
                len(skip_indices) / len(ranked_candidates)
                if ranked_candidates
                else 0.0
            ),
        },
        "ranked_candidates": ranked_candidates,
        "paths": {
            "full_graph": full_path,
            "compressed_graph": compressed_path,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
