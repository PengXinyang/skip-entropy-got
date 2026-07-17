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
}

REFINE_OPERATION_TYPES = {
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
    skip_thought_indices: Dict[int, set] = None,
    skip_refine_indices: Dict[int, set] = None,
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
        skip_thought_indices=skip_thought_indices or {},
        skip_refine_indices=skip_refine_indices or {},
    )
    ctrl.run()
    ctrl.output_graph(output_path)
    return ctrl, read_json(output_path)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def select_low_entropy_thoughts(
    full_graph_json: List[Dict[str, Any]],
    skip_ratio: float,
    entropy_field: str,
) -> Tuple[Dict[int, set], Dict[int, set], List[Dict[str, Any]]]:
    candidates = []
    refine_delta_field = (
        "abs_delta_normalized_avg_entropy_bits"
        if entropy_field == "normalized_avg_entropy_bits"
        else "abs_delta_entropy_bits"
    )
    for record in full_graph_json:
        if "operation" not in record:
            continue
        operation = record["operation"]
        # root generate 通常会创建初始计划或拆分结果。
        # 跳过它会破坏诸如 "part" 这样的任务特定字段，因此第一个静态实验保留它。
        if not record.get("predecessors"):
            continue
        if operation in REFINE_OPERATION_TYPES:
            for metadata in record.get("thought_metadata", []):
                for event in metadata.get(
                    "validate_and_improve_refine_events", []
                ):
                    delta = event.get(refine_delta_field)
                    if not isinstance(delta, (int, float)):
                        continue
                    candidates.append(
                        {
                            "candidate_type": "validate_and_improve_refine",
                            "node_label": (
                                f"op{record['operation_index']}:"
                                f"validate_and_improve:"
                                f"input{event['input_thought_index']}:"
                                f"try{event['try_index']}:refine"
                            ),
                            "operation_index": record["operation_index"],
                            "operation_id": record["operation_id"],
                            "operation": operation,
                            "predecessors": record.get("predecessors", []),
                            "successors": record.get("successors", []),
                            "input_thought_index": event["input_thought_index"],
                            "try_index": event["try_index"],
                            "entropy": float(delta),
                            "ranking_score": float(delta),
                            "delta_entropy_field": refine_delta_field,
                            "delta_entropy_bits": event.get("delta_entropy_bits"),
                            "abs_delta_entropy_bits": event.get(
                                "abs_delta_entropy_bits"
                            ),
                            "delta_normalized_avg_entropy_bits": event.get(
                                "delta_normalized_avg_entropy_bits"
                            ),
                            "abs_delta_normalized_avg_entropy_bits": event.get(
                                "abs_delta_normalized_avg_entropy_bits"
                            ),
                        }
                    )
            continue
        if operation not in SKIPPABLE_OPERATION_TYPES:
            continue
        for thought_index, metadata in enumerate(record.get("thought_metadata", [])):
            entropy = metadata.get(entropy_field)
            if not isinstance(entropy, (int, float)):
                continue
            thoughts = record.get("thoughts", [])
            thought_state = (
                thoughts[thought_index]
                if thought_index < len(thoughts)
                else {}
            )
            candidates.append(
                {
                    "candidate_type": "thought",
                    "node_label": (
                        f"op{record['operation_index']}:"
                        f"{operation}:thought{thought_index}"
                    ),
                    "operation_index": record["operation_index"],
                    "operation_id": record["operation_id"],
                    "operation": operation,
                    "predecessors": record.get("predecessors", []),
                    "successors": record.get("successors", []),
                    "thought_index": thought_index,
                    "entropy": float(entropy),
                    "ranking_score": float(entropy),
                    "entropy_field": entropy_field,
                    "thought_preview": str(
                        (thought_state or {}).get("current", "")
                    )[:200],
                }
            )

    candidates.sort(key=lambda item: item["entropy"])
    num_to_skip = int(len(candidates) * skip_ratio)
    selected = candidates[:num_to_skip]
    skip_thought_indices: Dict[int, set] = {}
    skip_refine_indices: Dict[int, set] = {}
    for item in selected:
        if item["candidate_type"] == "validate_and_improve_refine":
            skip_refine_indices.setdefault(item["operation_index"], set()).add(
                (item["input_thought_index"], item["try_index"])
            )
        else:
            skip_thought_indices.setdefault(item["operation_index"], set()).add(
                item["thought_index"]
            )
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
        if item["candidate_type"] == "validate_and_improve_refine":
            item["selected_for_skip"] = (
                item["input_thought_index"],
                item["try_index"],
            ) in skip_refine_indices.get(item["operation_index"], set())
        else:
            item["selected_for_skip"] = item["thought_index"] in (
                skip_thought_indices.get(item["operation_index"], set())
            )
    return skip_thought_indices, skip_refine_indices, candidates


def write_candidate_ranking(
    ranked_candidates: List[Dict[str, Any]],
    json_path: str,
    csv_path: str,
) -> None:
    """
    输出可跳过节点的熵/ΔH 排名。

    JSON 保留完整字段；CSV 只保留最常用的分析列，便于快速查看和画图。
    """
    with open(json_path, "w") as f:
        json.dump(ranked_candidates, f, indent=2)

    fieldnames = [
        "rank",
        "selected_for_skip",
        "candidate_type",
        "node_label",
        "operation_index",
        "operation_id",
        "operation",
        "thought_index",
        "input_thought_index",
        "try_index",
        "ranking_score",
        "entropy",
        "entropy_field",
        "delta_entropy_field",
        "delta_entropy_bits",
        "abs_delta_entropy_bits",
        "delta_normalized_avg_entropy_bits",
        "abs_delta_normalized_avg_entropy_bits",
        "predecessors",
        "successors",
        "thought_preview",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in ranked_candidates:
            row = {field: candidate.get(field) for field in fieldnames}
            row["predecessors"] = json.dumps(row["predecessors"])
            row["successors"] = json.dumps(row["successors"])
            writer.writerow(row)


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
        description="Run a static low-entropy thought-level [SKIP] replay experiment on sorting_032 GoT."
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
    ranking_json_path = os.path.join(run_dir, "candidate_ranking.json")
    ranking_csv_path = os.path.join(run_dir, "candidate_ranking.csv")

    _, full_json = run_sorting_got(
        args.config_path,
        args.model_name,
        problem,
        full_path,
    )
    (
        skip_thought_indices,
        skip_refine_indices,
        ranked_candidates,
    ) = select_low_entropy_thoughts(
        full_json,
        args.skip_ratio,
        args.entropy_field,
    )
    write_candidate_ranking(
        ranked_candidates,
        ranking_json_path,
        ranking_csv_path,
    )
    _, compressed_json = run_sorting_got(
        args.config_path,
        args.model_name,
        problem,
        compressed_path,
        skip_thought_indices=skip_thought_indices,
        skip_refine_indices=skip_refine_indices,
    )

    full_tokens = token_summary(full_json)
    compressed_tokens = token_summary(compressed_json)
    summary = {
        "task": "sorting_032_got",
        "data_id": args.data_id,
        "model_name": args.model_name,
        "entropy_field": args.entropy_field,
        "skip_ratio": args.skip_ratio,
        "selected_skip_thought_indices": {
            str(operation_index): sorted(thought_indices)
            for operation_index, thought_indices in sorted(
                skip_thought_indices.items()
            )
        },
        "selected_skip_refine_indices": {
            str(operation_index): [
                {
                    "input_thought_index": input_thought_index,
                    "try_index": try_index,
                }
                for input_thought_index, try_index in sorted(refine_indices)
            ]
            for operation_index, refine_indices in sorted(
                skip_refine_indices.items()
            )
        },
        "num_candidates": len(ranked_candidates),
        "num_skipped_thoughts": sum(
            len(thought_indices)
            for thought_indices in skip_thought_indices.values()
        ),
        "num_skipped_refines": sum(
            len(refine_indices)
            for refine_indices in skip_refine_indices.values()
        ),
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
            "thought_skip_ratio": (
                (
                    sum(
                        len(thought_indices)
                        for thought_indices in skip_thought_indices.values()
                    )
                    + sum(
                        len(refine_indices)
                        for refine_indices in skip_refine_indices.values()
                    )
                )
                / len(ranked_candidates)
                if ranked_candidates
                else 0.0
            ),
        },
        "ranked_candidates": ranked_candidates,
        "paths": {
            "full_graph": full_path,
            "compressed_graph": compressed_path,
            "candidate_ranking_json": ranking_json_path,
            "candidate_ranking_csv": ranking_csv_path,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
