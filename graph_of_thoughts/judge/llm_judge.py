import csv
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from graph_of_thoughts.language_models import AbstractLanguageModel


JUDGE_SKIPPABLE_OPERATIONS = {"generate", "aggregate", "improve"}


@dataclass
class JudgeResult:
    candidate: Dict[str, Any]
    scores: Dict[str, Any]
    raw_response: str
    judge_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        row = dict(self.candidate)
        row["judge"] = self.scores
        row["judge_raw_response"] = self.raw_response
        row["judge_metadata"] = self.judge_metadata
        return row


def _shorten(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _current_text(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("current", ""))
    return str(state)


def _find_final_output(graph_json: List[Dict[str, Any]]) -> str:
    for record in reversed(graph_json):
        if not isinstance(record, dict) or "thoughts" not in record:
            continue
        thoughts = record.get("thoughts") or []
        for thought in reversed(thoughts):
            current = _current_text(thought).strip()
            if current and current != "[SKIP]":
                return current
    return ""


def _operation_by_id(graph_json: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    return {
        record.get("operation_id"): record
        for record in graph_json
        if isinstance(record, dict) and "operation_id" in record
    }


def next_keep_best_n_by_operation(
    graph_json: List[Dict[str, Any]]
) -> Dict[int, int]:
    records_by_id = _operation_by_id(graph_json)
    result: Dict[int, int] = {}
    for record in graph_json:
        if not isinstance(record, dict) or "operation" not in record:
            continue
        operation_index = record.get("operation_index")
        queue = list(record.get("successors", []) or [])
        visited = set()
        while queue:
            successor_id = queue.pop(0)
            if successor_id in visited:
                continue
            visited.add(successor_id)
            successor = records_by_id.get(successor_id)
            if not successor:
                continue
            if successor.get("operation") == "keep_best_n":
                result[operation_index] = max(
                    1,
                    len(successor.get("thoughts", []) or []),
                )
                break
            queue.extend(successor.get("successors", []) or [])
    return result


def min_remaining_candidates_for_operation(
    operation: str,
    operation_index: int,
    keep_best_n_by_operation: Dict[int, int],
) -> int:
    if operation == "aggregate":
        return max(1, keep_best_n_by_operation.get(operation_index, 1))
    return 1


def _predecessor_outputs(
    record: Dict[str, Any],
    operations_by_id: Dict[Any, Dict[str, Any]],
    max_chars_per_predecessor: int,
) -> List[Dict[str, Any]]:
    outputs = []
    for predecessor_id in record.get("predecessors", []) or []:
        predecessor = operations_by_id.get(predecessor_id)
        if not predecessor:
            continue
        thoughts = predecessor.get("thoughts") or []
        outputs.append(
            {
                "operation_id": predecessor.get("operation_id"),
                "operation_index": predecessor.get("operation_index"),
                "operation": predecessor.get("operation"),
                "outputs": [
                    _shorten(_current_text(thought), max_chars_per_predecessor)
                    for thought in thoughts
                ],
            }
        )
    return outputs


def collect_thought_judge_candidates(
    graph_json: List[Dict[str, Any]],
    *,
    entropy_field: str = "normalized_avg_entropy_bits",
    include_root: bool = False,
    include_operations: Optional[Iterable[str]] = None,
    max_text_chars: int = 2000,
    max_predecessor_chars: int = 800,
) -> List[Dict[str, Any]]:
    """
    从 full_graph.json 中抽取需要 judge 的 thought 节点。

    默认只抽取当前 skip 实验真正可能跳过的 generate/aggregate/improve thought。
    score、selector、keep_best、ground_truth 本身不调用大模型生成新 thought，
    因此不作为第一版 judge-skip 的候选。
    """
    allowed_operations = set(include_operations or JUDGE_SKIPPABLE_OPERATIONS)
    operations_by_id = _operation_by_id(graph_json)
    keep_best_n_by_operation = next_keep_best_n_by_operation(graph_json)
    final_output = _find_final_output(graph_json)
    candidates: List[Dict[str, Any]] = []

    for record in graph_json:
        if not isinstance(record, dict) or "operation" not in record:
            continue
        operation = record.get("operation")
        if operation not in allowed_operations:
            continue
        if not include_root and not record.get("predecessors"):
            continue

        thoughts = record.get("thoughts") or []
        metadata_items = record.get("thought_metadata") or []
        for thought_index, thought_state in enumerate(thoughts):
            current = _current_text(thought_state).strip()
            if not current or current == "[SKIP]":
                continue
            metadata = (
                metadata_items[thought_index]
                if thought_index < len(metadata_items)
                and isinstance(metadata_items[thought_index], dict)
                else {}
            )
            candidate = {
                "candidate_type": "thought",
                "node_label": (
                    f"op{record.get('operation_index')}:"
                    f"{operation}:thought{thought_index}"
                ),
                "operation_id": record.get("operation_id"),
                "operation_index": record.get("operation_index"),
                "operation": operation,
                "thought_index": thought_index,
                "min_remaining_candidates": min_remaining_candidates_for_operation(
                    operation,
                    record.get("operation_index"),
                    keep_best_n_by_operation,
                ),
                "predecessors": record.get("predecessors", []),
                "successors": record.get("successors", []),
                "score": (
                    record.get("scores", [None] * len(thoughts))[thought_index]
                    if thought_index < len(record.get("scores", []))
                    else None
                ),
                "entropy_field": entropy_field,
                "entropy": metadata.get(entropy_field),
                "avg_entropy_bits": metadata.get("avg_entropy_bits"),
                "normalized_avg_entropy_bits": metadata.get(
                    "normalized_avg_entropy_bits"
                ),
                "prompt_role": metadata.get("prompt_role"),
                "response_text": _shorten(
                    metadata.get("response_text", current),
                    max_text_chars,
                ),
                "thought_state": thought_state,
                "thought_current": _shorten(current, max_text_chars),
                "predecessor_outputs": _predecessor_outputs(
                    record,
                    operations_by_id,
                    max_predecessor_chars,
                ),
                "final_output": _shorten(final_output, max_text_chars),
            }
            candidates.append(candidate)

    return candidates


def parse_judge_json(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {
        "task_relevance": None,
        "input_output_consistency": None,
        "final_answer_contribution": None,
        "redundancy": None,
        "skip_risk": None,
        "usefulness": None,
        "parse_failed": True,
        "reason": stripped[:500],
    }


def normalize_judge_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(scores)
    for field in (
        "task_relevance",
        "input_output_consistency",
        "final_answer_contribution",
        "redundancy",
        "skip_risk",
    ):
        value = normalized.get(field)
        if isinstance(value, (int, float)):
            normalized[field] = max(0.0, min(1.0, float(value)))
        else:
            normalized[field] = None

    if not isinstance(normalized.get("usefulness"), (int, float)):
        required = [
            normalized.get("task_relevance"),
            normalized.get("input_output_consistency"),
            normalized.get("final_answer_contribution"),
            normalized.get("redundancy"),
        ]
        if all(isinstance(value, (int, float)) for value in required):
            normalized["usefulness"] = (
                0.3 * normalized["task_relevance"]
                + 0.3 * normalized["input_output_consistency"]
                + 0.3 * normalized["final_answer_contribution"]
                - 0.1 * normalized["redundancy"]
            )
        else:
            normalized["usefulness"] = None
    else:
        normalized["usefulness"] = max(0.0, min(1.0, float(normalized["usefulness"])))
    return normalized


class LLMThoughtJudge:
    def __init__(self, lm: AbstractLanguageModel, prompt_builder: Any) -> None:
        self.lm = lm
        self.prompt_builder = prompt_builder

    def judge(self, task_context: Dict[str, Any], candidate: Dict[str, Any]) -> JudgeResult:
        prompt = self.prompt_builder.build_prompt(task_context, candidate)
        responses = self.lm.get_response_texts(self.lm.query(prompt, num_responses=1))
        metadata = self.lm.consume_last_response_metadata(1)
        raw_response = responses[0] if responses else ""
        scores = normalize_judge_scores(parse_judge_json(raw_response))
        scores["judge_prompt"] = prompt
        return JudgeResult(
            candidate=candidate,
            scores=scores,
            raw_response=raw_response,
            judge_metadata=metadata[0] if metadata else {},
        )

    def judge_all(
        self,
        task_context: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        *,
        max_candidates: Optional[int] = None,
    ) -> List[JudgeResult]:
        selected = candidates[:max_candidates] if max_candidates else candidates
        return [self.judge(task_context, candidate) for candidate in selected]


def judge_summary(results: List[JudgeResult]) -> Dict[str, Any]:
    rows = [result.to_dict() for result in results]
    numeric_fields = [
        "task_relevance",
        "input_output_consistency",
        "final_answer_contribution",
        "redundancy",
        "skip_risk",
        "usefulness",
    ]
    summary: Dict[str, Any] = {
        "num_judged_candidates": len(rows),
        "num_parse_failed": sum(
            1 for row in rows if row.get("judge", {}).get("parse_failed")
        ),
    }
    for field in numeric_fields:
        values = [
            row["judge"][field]
            for row in rows
            if isinstance(row.get("judge", {}).get(field), (int, float))
        ]
        if not values:
            summary[field] = None
            continue
        summary[field] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    low_risk = [
        row
        for row in rows
        if isinstance(row.get("judge", {}).get("skip_risk"), (int, float))
    ]
    low_risk.sort(key=lambda row: (row["judge"]["skip_risk"], row["judge"].get("usefulness") or 0.0))
    summary["lowest_skip_risk_nodes"] = [
        {
            "node_label": row.get("node_label"),
            "operation": row.get("operation"),
            "entropy": row.get("entropy"),
            "skip_risk": row["judge"].get("skip_risk"),
            "usefulness": row["judge"].get("usefulness"),
            "reason": row["judge"].get("reason"),
        }
        for row in low_risk[:20]
    ]
    return summary


def select_judge_skip_recommendations(
    results: List[JudgeResult],
    skip_ratio: float,
) -> List[Dict[str, Any]]:
    rows = [result.to_dict() for result in results]
    candidate_counts: Dict[Any, int] = {}
    for row in rows:
        operation_index = row.get("operation_index")
        candidate_counts[operation_index] = candidate_counts.get(operation_index, 0) + 1

    skippable_rows = []
    for row in rows:
        operation_index = row.get("operation_index")
        min_remaining = int(row.get("min_remaining_candidates") or 1)
        if candidate_counts.get(operation_index, 0) <= min_remaining:
            row["judge_skip_exclusion_reason"] = (
                "not_enough_candidates_after_min_remaining_constraint"
            )
            continue
        skippable_rows.append(row)

    def sort_key(row: Dict[str, Any]) -> tuple:
        judge = row.get("judge", {})
        skip_risk = judge.get("skip_risk")
        usefulness = judge.get("usefulness")
        contribution = judge.get("final_answer_contribution")
        redundancy = judge.get("redundancy")
        return (
            skip_risk if isinstance(skip_risk, (int, float)) else 1.0,
            usefulness if isinstance(usefulness, (int, float)) else 1.0,
            contribution if isinstance(contribution, (int, float)) else 1.0,
            -(redundancy if isinstance(redundancy, (int, float)) else 0.0),
        )

    skippable_rows.sort(key=sort_key)
    num_to_skip = int(len(skippable_rows) * skip_ratio)
    selected_counts: Dict[Any, int] = {}
    selected_keys = set()
    ranked_rows = []

    for rank, row in enumerate(skippable_rows, start=1):
        row = dict(row)
        row["judge_skip_rank"] = rank
        operation_index = row.get("operation_index")
        selected_count = selected_counts.get(operation_index, 0)
        total_count = candidate_counts.get(operation_index, 0)
        min_remaining = int(row.get("min_remaining_candidates") or 1)
        can_select = selected_count < total_count - min_remaining
        if len(selected_keys) < num_to_skip and can_select:
            row["selected_for_judge_skip"] = True
            selected_counts[operation_index] = selected_count + 1
            selected_keys.add((operation_index, row.get("thought_index")))
        else:
            row["selected_for_judge_skip"] = False
            if not can_select:
                row["judge_skip_exclusion_reason"] = (
                    "keep_required_thought_candidates_per_operation"
                )
        ranked_rows.append(row)

    return ranked_rows


def write_judge_outputs(
    results: List[JudgeResult],
    *,
    json_path: str,
    csv_path: str,
    summary_path: str,
    recommendations_json_path: Optional[str] = None,
    recommendations_csv_path: Optional[str] = None,
    skip_ratio: Optional[float] = None,
) -> None:
    rows = [result.to_dict() for result in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "node_label",
        "operation_index",
        "operation_id",
        "operation",
        "thought_index",
        "min_remaining_candidates",
        "entropy",
        "avg_entropy_bits",
        "normalized_avg_entropy_bits",
        "score",
        "task_relevance",
        "input_output_consistency",
        "final_answer_contribution",
        "redundancy",
        "skip_risk",
        "usefulness",
        "parse_failed",
        "reason",
        "thought_current",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            judge = row.get("judge", {})
            writer.writerow(
                {
                    "node_label": row.get("node_label"),
                    "operation_index": row.get("operation_index"),
                    "operation_id": row.get("operation_id"),
                    "operation": row.get("operation"),
                    "thought_index": row.get("thought_index"),
                    "min_remaining_candidates": row.get(
                        "min_remaining_candidates"
                    ),
                    "entropy": row.get("entropy"),
                    "avg_entropy_bits": row.get("avg_entropy_bits"),
                    "normalized_avg_entropy_bits": row.get(
                        "normalized_avg_entropy_bits"
                    ),
                    "score": row.get("score"),
                    "task_relevance": judge.get("task_relevance"),
                    "input_output_consistency": judge.get(
                        "input_output_consistency"
                    ),
                    "final_answer_contribution": judge.get(
                        "final_answer_contribution"
                    ),
                    "redundancy": judge.get("redundancy"),
                    "skip_risk": judge.get("skip_risk"),
                    "usefulness": judge.get("usefulness"),
                    "parse_failed": judge.get("parse_failed", False),
                    "reason": judge.get("reason"),
                    "thought_current": row.get("thought_current"),
                }
            )

    with open(summary_path, "w", encoding="utf-8") as f:
        summary = judge_summary(results)
        if (
            recommendations_json_path
            and recommendations_csv_path
            and isinstance(skip_ratio, (int, float))
        ):
            recommendations = select_judge_skip_recommendations(results, skip_ratio)
            summary["judge_skip_ratio"] = skip_ratio
            summary["num_judge_skip_candidates"] = len(recommendations)
            summary["num_selected_for_judge_skip"] = sum(
                1 for row in recommendations if row.get("selected_for_judge_skip")
            )
            with open(recommendations_json_path, "w", encoding="utf-8") as rec_json:
                json.dump(recommendations, rec_json, indent=2, ensure_ascii=False)

            recommendation_fields = [
                "judge_skip_rank",
                "selected_for_judge_skip",
                "judge_skip_exclusion_reason",
                "node_label",
                "operation_index",
                "operation_id",
                "operation",
                "thought_index",
                "min_remaining_candidates",
                "entropy",
                "score",
                "task_relevance",
                "input_output_consistency",
                "final_answer_contribution",
                "redundancy",
                "skip_risk",
                "usefulness",
                "reason",
                "thought_current",
            ]
            with open(
                recommendations_csv_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as rec_csv:
                writer = csv.DictWriter(rec_csv, fieldnames=recommendation_fields)
                writer.writeheader()
                for row in recommendations:
                    judge = row.get("judge", {})
                    writer.writerow(
                        {
                            "judge_skip_rank": row.get("judge_skip_rank"),
                            "selected_for_judge_skip": row.get(
                                "selected_for_judge_skip"
                            ),
                            "judge_skip_exclusion_reason": row.get(
                                "judge_skip_exclusion_reason"
                            ),
                            "node_label": row.get("node_label"),
                            "operation_index": row.get("operation_index"),
                            "operation_id": row.get("operation_id"),
                            "operation": row.get("operation"),
                            "thought_index": row.get("thought_index"),
                            "min_remaining_candidates": row.get(
                                "min_remaining_candidates"
                            ),
                            "entropy": row.get("entropy"),
                            "score": row.get("score"),
                            "task_relevance": judge.get("task_relevance"),
                            "input_output_consistency": judge.get(
                                "input_output_consistency"
                            ),
                            "final_answer_contribution": judge.get(
                                "final_answer_contribution"
                            ),
                            "redundancy": judge.get("redundancy"),
                            "skip_risk": judge.get("skip_risk"),
                            "usefulness": judge.get("usefulness"),
                            "reason": judge.get("reason"),
                            "thought_current": row.get("thought_current"),
                        }
                    )
        json.dump(summary, f, indent=2, ensure_ascii=False)
