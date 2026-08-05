import argparse
import csv
import datetime
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_of_thoughts import controller
from graph_of_thoughts.language_models.factory import build_language_model

from examples.doc_merge import doc_merge
from examples.keyword_counting import keyword_counting
from examples.set_intersection import (
    set_intersection_032,
    set_intersection_064,
    set_intersection_128,
)
from examples.static_skip_experiment import (
    final_solved,
    reduction_ratio,
    select_low_entropy_thoughts,
    token_summary,
    write_candidate_ranking,
)


Case = Dict[str, Any]


@dataclass
class TaskSpec:
    # name: 输出目录和 summary 中使用的任务名。
    name: str
    # method_name: GoT 构图方法名，会写入初始 state["method"]，影响 prompter 分支。
    method_name: str
    # load_cases: 从任务数据集读取样例，返回统一的 case 字典。
    load_cases: Callable[[Optional[Sequence[int]], Optional[int]], List[Case]]
    # build_graph: 构造 GraphOfOperations；关键词统计需要额外传 all_potential_countries。
    build_graph: Callable[[], Any]
    # build_prompter/build_parser: 对应任务自己的 prompt 和 parse 逻辑。
    build_prompter: Callable[[], Any]
    build_parser: Callable[[], Any]
    # initial_state: 将 case 转成 Controller 需要的初始 thought state。
    initial_state: Callable[[Case, str], Dict[str, Any]]
    # has_ground_truth: doc_merge 原任务没有 GroundTruth operation，不能用 solved 评价正确率。
    has_ground_truth: bool = True


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def limit_cases(cases: List[Case], max_cases: Optional[int]) -> List[Case]:
    if max_cases is None:
        return cases
    return cases[:max_cases]


def load_doc_merge_cases(
    data_ids: Optional[Sequence[int]], max_cases: Optional[int]
) -> List[Case]:
    path = os.path.join(os.path.dirname(__file__), "doc_merge", "documents.csv")
    cases: List[Case] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            case_id = int(row[0])
            if data_ids is not None and case_id not in data_ids:
                continue
            cases.append(
                {
                    "id": case_id,
                    "problem": row[1],
                    "documents": [row[2], row[3], row[4], row[5]],
                }
            )
    return limit_cases(cases, max_cases)


def load_keyword_counting_cases(
    data_ids: Optional[Sequence[int]], max_cases: Optional[int]
) -> List[Case]:
    path = os.path.join(
        os.path.dirname(__file__), "keyword_counting", "countries.csv"
    )
    cases: List[Case] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            case_id = int(row[0])
            if data_ids is not None and case_id not in data_ids:
                continue
            cases.append(
                {
                    "id": case_id,
                    "original": row[1],
                    "ground_truth": row[2],
                }
            )
    return limit_cases(cases, max_cases)


def load_all_potential_countries() -> List[str]:
    cases = load_keyword_counting_cases(None, None)
    countries = set()
    for case in cases:
        ground_truth = case["ground_truth"].strip()
        for country in ground_truth[1:-1].split(", "):
            if country:
                countries.add(country)
    return sorted(countries)


def load_set_intersection_cases(
    dataset_name: str,
    data_ids: Optional[Sequence[int]],
    max_cases: Optional[int],
) -> List[Case]:
    path = os.path.join(
        os.path.dirname(__file__),
        "set_intersection",
        f"{dataset_name}.csv",
    )
    cases: List[Case] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            case_id = int(row[0])
            if data_ids is not None and case_id not in data_ids:
                continue
            cases.append(
                {
                    "id": case_id,
                    "set1": row[1],
                    "set2": row[2],
                    "result": row[3],
                }
            )
    return limit_cases(cases, max_cases)


def make_jsonable(value: Any) -> Any:
    # Controller.output_graph 直接 json.dumps(thought.state)，doc_merge 的 parts 是 set，
    # 因此输出前统一转换为 JSON 可序列化结构。
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]
    return value


def sanitize_graph_for_json(graph: Any) -> None:
    for operation in graph.operations:
        for thought in operation.thoughts:
            thought.state = make_jsonable(thought.state)
            thought.metadata = make_jsonable(thought.metadata)


def run_task_got(
    task: TaskSpec,
    config_path: str,
    model_name: str,
    case: Case,
    output_path: str,
    skip_thought_indices: Optional[Dict[int, set]] = None,
    skip_refine_indices: Optional[Dict[int, set]] = None,
) -> Tuple[controller.Controller, List[Dict[str, Any]]]:
    lm = build_language_model(config_path, model_name, cache=False)
    graph = task.build_graph()
    ctrl = controller.Controller(
        lm,
        graph,
        task.build_prompter(),
        task.build_parser(),
        task.initial_state(case, task.method_name),
        skip_thought_indices=skip_thought_indices or {},
        skip_refine_indices=skip_refine_indices or {},
    )
    ctrl.run()
    sanitize_graph_for_json(graph)
    ctrl.output_graph(output_path)
    return ctrl, read_json(output_path)


def case_summary(
    task: TaskSpec,
    case: Case,
    model_name: str,
    entropy_field: str,
    skip_ratio: float,
    skip_order: str,
    full_json: List[Dict[str, Any]],
    compressed_json: List[Dict[str, Any]],
    ranked_candidates: List[Dict[str, Any]],
    skip_thought_indices: Dict[int, set],
    skip_refine_indices: Dict[int, set],
    paths: Dict[str, str],
) -> Dict[str, Any]:
    full_tokens = token_summary(full_json)
    compressed_tokens = token_summary(compressed_json)
    skipped_thoughts = sum(len(items) for items in skip_thought_indices.values())
    skipped_refines = sum(len(items) for items in skip_refine_indices.values())
    return {
        "task": task.name,
        "method": task.method_name,
        "case_id": case["id"],
        "model_name": model_name,
        "entropy_field": entropy_field,
        "skip_ratio": skip_ratio,
        "skip_order": skip_order,
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
        "num_skipped_thoughts": skipped_thoughts,
        "num_skipped_refines": skipped_refines,
        "full": {
            "solved": final_solved(full_json) if task.has_ground_truth else None,
            **full_tokens,
        },
        "compressed": {
            "solved": (
                final_solved(compressed_json) if task.has_ground_truth else None
            ),
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
                (skipped_thoughts + skipped_refines) / len(ranked_candidates)
                if ranked_candidates
                else 0.0
            ),
        },
        "paths": paths,
    }


def build_task_specs() -> Dict[str, TaskSpec]:
    all_potential_countries = load_all_potential_countries()
    return {
        "doc_merge": TaskSpec(
            name="doc_merge",
            method_name="got2",
            load_cases=load_doc_merge_cases,
            build_graph=doc_merge.got2,
            build_prompter=doc_merge.DocMergePrompter,
            build_parser=doc_merge.DocMergeParser,
            initial_state=lambda case, method_name: {
                "documents": case["documents"],
                "parts": set(),
                "current": "",
                "method": method_name,
            },
            has_ground_truth=False,
        ),
        "keyword_counting": TaskSpec(
            name="keyword_counting",
            method_name="got4",
            load_cases=load_keyword_counting_cases,
            build_graph=lambda: keyword_counting.got4(all_potential_countries),
            build_prompter=keyword_counting.KeywordCountingPrompter,
            build_parser=keyword_counting.KeywordCountingParser,
            initial_state=lambda case, method_name: {
                "original": case["original"],
                "ground_truth": case["ground_truth"],
                "current": "",
                "phase": 0,
                "method": method_name,
            },
        ),
        "set_intersection_032": TaskSpec(
            name="set_intersection_032",
            method_name="got",
            load_cases=lambda data_ids, max_cases: load_set_intersection_cases(
                "set_intersection_032", data_ids, max_cases
            ),
            build_graph=set_intersection_032.got,
            build_prompter=set_intersection_032.SetIntersectionPrompter,
            build_parser=set_intersection_032.SetIntersectionParser,
            initial_state=lambda case, method_name: {
                "set1": case["set1"],
                "set2": case["set2"],
                "result": case["result"],
                "current": "",
                "phase": 0,
                "method": method_name,
            },
        ),
        "set_intersection_064": TaskSpec(
            name="set_intersection_064",
            method_name="got",
            load_cases=lambda data_ids, max_cases: load_set_intersection_cases(
                "set_intersection_064", data_ids, max_cases
            ),
            build_graph=set_intersection_064.got,
            build_prompter=set_intersection_064.SetIntersectionPrompter,
            build_parser=set_intersection_064.SetIntersectionParser,
            initial_state=lambda case, method_name: {
                "set1": case["set1"],
                "set2": case["set2"],
                "result": case["result"],
                "current": "",
                "phase": 0,
                "method": method_name,
            },
        ),
        "set_intersection_128": TaskSpec(
            name="set_intersection_128",
            method_name="got",
            load_cases=lambda data_ids, max_cases: load_set_intersection_cases(
                "set_intersection_128", data_ids, max_cases
            ),
            build_graph=set_intersection_128.got,
            build_prompter=set_intersection_128.SetIntersectionPrompter,
            build_parser=set_intersection_128.SetIntersectionParser,
            initial_state=lambda case, method_name: {
                "set1": case["set1"],
                "set2": case["set2"],
                "result": case["result"],
                "current": "",
                "phase": 0,
                "method": method_name,
            },
        ),
    }


def selected_tasks(
    task_specs: Dict[str, TaskSpec], requested: Optional[str]
) -> List[TaskSpec]:
    default_tasks = ["doc_merge", "keyword_counting", "set_intersection_032"]
    names = (
        [item.strip() for item in requested.split(",") if item.strip()]
        if requested
        else default_tasks
    )
    unknown = [name for name in names if name not in task_specs]
    if unknown:
        raise ValueError(
            f"Unknown task(s): {unknown}. Available tasks: {sorted(task_specs)}"
        )
    return [task_specs[name] for name in names]


def run_case_orders(
    task: TaskSpec,
    case: Case,
    args: argparse.Namespace,
    run_root: str,
) -> List[Dict[str, Any]]:
    case_dir = os.path.join(run_root, task.name, f"id{case['id']}")
    os.makedirs(case_dir, exist_ok=True)

    full_path = os.path.join(case_dir, "full_graph.json")
    _, full_json = run_task_got(
        task,
        args.config_path,
        args.model_name,
        case,
        full_path,
    )

    summaries = []
    for skip_order in args.skip_orders:
        order_dir = os.path.join(case_dir, skip_order)
        os.makedirs(order_dir, exist_ok=True)
        ranking_json_path = os.path.join(order_dir, "candidate_ranking.json")
        ranking_csv_path = os.path.join(order_dir, "candidate_ranking.csv")
        compressed_path = os.path.join(order_dir, "compressed_graph.json")
        summary_path = os.path.join(order_dir, "summary.json")

        (
            skip_thought_indices,
            skip_refine_indices,
            ranked_candidates,
        ) = select_low_entropy_thoughts(
            full_json,
            args.skip_ratio,
            args.entropy_field,
            skip_order,
        )
        write_candidate_ranking(
            ranked_candidates,
            ranking_json_path,
            ranking_csv_path,
        )
        _, compressed_json = run_task_got(
            task,
            args.config_path,
            args.model_name,
            case,
            compressed_path,
            skip_thought_indices=skip_thought_indices,
            skip_refine_indices=skip_refine_indices,
        )

        summary = case_summary(
            task,
            case,
            args.model_name,
            args.entropy_field,
            args.skip_ratio,
            skip_order,
            full_json,
            compressed_json,
            ranked_candidates,
            skip_thought_indices,
            skip_refine_indices,
            {
                "full_graph": full_path,
                "compressed_graph": compressed_path,
                "candidate_ranking_json": ranking_json_path,
                "candidate_ranking_csv": ranking_csv_path,
                "summary": summary_path,
            },
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)
    return summaries


def aggregate_batch_summary(
    summaries: Iterable[Dict[str, Any]],
    paths: Dict[str, str],
) -> Dict[str, Any]:
    rows = list(summaries)
    solved_rows = [
        row
        for row in rows
        if row["full"]["solved"] is not None
        and row["compressed"]["solved"] is not None
    ]
    return {
        "num_runs": len(rows),
        "num_evaluable_runs": len(solved_rows),
        "full_accuracy": (
            sum(1 for row in solved_rows if row["full"]["solved"])
            / len(solved_rows)
            if solved_rows
            else None
        ),
        "compressed_accuracy": (
            sum(1 for row in solved_rows if row["compressed"]["solved"])
            / len(solved_rows)
            if solved_rows
            else None
        ),
        "avg_total_token_reduction": (
            sum(row["reductions"]["total_token_reduction"] for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "avg_api_call_reduction": (
            sum(row["reductions"]["api_call_reduction"] for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "avg_cost_reduction": (
            sum(row["reductions"]["cost_reduction"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "runs": rows,
        "paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch static thought-level [SKIP] replay experiment for doc_merge, "
            "keyword_counting, and set_intersection tasks."
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
    parser.add_argument(
        "--tasks",
        default=None,
        help=(
            "Comma-separated task names. Defaults to "
            "doc_merge,keyword_counting,set_intersection_032."
        ),
    )
    parser.add_argument(
        "--data-ids",
        default=None,
        help="Comma-separated case ids, for example 0,1,2. Defaults to dataset order.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=1,
        help="Maximum cases per task when --data-ids is not enough. Default: 1.",
    )
    parser.add_argument("--skip-ratio", type=float, default=0.2)
    parser.add_argument(
        "--skip-orders",
        nargs="+",
        choices=["low", "high"],
        default=["low", "high"],
        help="Run low-entropy skip, high-entropy skip, or both. Default: both.",
    )
    parser.add_argument("--entropy-field", default="normalized_avg_entropy_bits")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "batch_static_skip_results"),
    )
    args = parser.parse_args()

    if not 0.0 <= args.skip_ratio <= 1.0:
        raise ValueError("--skip-ratio must be between 0 and 1")
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")

    task_specs = build_task_specs()
    tasks = selected_tasks(task_specs, args.tasks)
    data_ids = parse_ids(args.data_ids)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = os.path.join(
        args.output_dir,
        (
            f"{args.model_name}_skip{args.skip_ratio}_"
            f"{args.entropy_field}_{timestamp}"
        ),
    )
    os.makedirs(run_root, exist_ok=True)

    all_summaries: List[Dict[str, Any]] = []
    for task in tasks:
        cases = task.load_cases(data_ids, args.max_cases)
        for case in cases:
            print(f"Running {task.name} id={case['id']}")
            all_summaries.extend(run_case_orders(task, case, args, run_root))

    batch_summary_path = os.path.join(run_root, "batch_summary.json")
    batch_summary = aggregate_batch_summary(
        all_summaries,
        {"run_root": run_root, "batch_summary": batch_summary_path},
    )
    with open(batch_summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)

    print(json.dumps(batch_summary, indent=2))


if __name__ == "__main__":
    main()
