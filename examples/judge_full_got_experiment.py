import argparse
import contextlib
import datetime
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_of_thoughts.judge import (
    LLMThoughtJudge,
    collect_thought_judge_candidates,
    write_judge_outputs,
)
from graph_of_thoughts.language_models.factory import build_language_model

from examples.batch_static_skip_experiment import (
    TaskSpec,
    build_task_specs,
    parse_ids,
    run_task_got,
    selected_tasks,
    token_summary,
)
from examples.task_judges import build_task_judge_prompt


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def limit_cases(cases: List[Dict[str, Any]], max_cases: Optional[int]) -> List[Dict[str, Any]]:
    if max_cases is None:
        return cases
    return cases[:max_cases]


def existing_full_graph_path(
    input_run_root: Optional[str],
    task_name: str,
    case_id: int,
) -> Optional[str]:
    if not input_run_root:
        return None

    candidates = [
        os.path.join(input_run_root, f"id{case_id}", "full_graph.json"),
        os.path.join(input_run_root, task_name, f"id{case_id}", "full_graph.json"),
    ]
    for name in os.listdir(input_run_root) if os.path.isdir(input_run_root) else []:
        candidates.append(
            os.path.join(input_run_root, name, f"id{case_id}", "full_graph.json")
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def run_full_graph_if_needed(
    task: TaskSpec,
    case: Dict[str, Any],
    args: argparse.Namespace,
    case_dir: str,
) -> List[Dict[str, Any]]:
    full_graph_path = (
        args.full_graph_path
        or existing_full_graph_path(args.input_run_root, task.name, case["id"])
        or os.path.join(case_dir, "full_graph.json")
    )
    if args.full_graph_path or args.input_run_root:
        if not os.path.exists(full_graph_path):
            raise FileNotFoundError(
                f"Could not find full_graph.json for {task.name} id={case['id']} "
                f"under {args.input_run_root}"
            )
        return read_json(full_graph_path)
    run_task_got(
        task,
        args.config_path,
        args.model_name,
        case,
        full_graph_path,
    )
    return read_json(full_graph_path)


def run_case_judge(
    task: TaskSpec,
    case: Dict[str, Any],
    args: argparse.Namespace,
    run_root: str,
) -> Dict[str, Any]:
    case_dir = os.path.join(run_root, f"id{case['id']}")
    os.makedirs(case_dir, exist_ok=True)

    full_graph_path = (
        args.full_graph_path
        or existing_full_graph_path(args.input_run_root, task.name, case["id"])
        or os.path.join(case_dir, "full_graph.json")
    )
    judge_json_path = os.path.join(case_dir, "judge_scores.json")
    judge_csv_path = os.path.join(case_dir, "judge_scores.csv")
    judge_summary_path = os.path.join(case_dir, "judge_summary.json")
    error_path = os.path.join(case_dir, "error.log")

    full_json = run_full_graph_if_needed(task, case, args, case_dir)
    candidates = collect_thought_judge_candidates(
        full_json,
        entropy_field=args.entropy_field,
        include_root=args.include_root,
        max_text_chars=args.max_text_chars,
        max_predecessor_chars=args.max_predecessor_chars,
    )
    prompt_builder = build_task_judge_prompt(task.name)
    judge_lm = build_language_model(
        args.config_path,
        args.judge_model_name or args.model_name,
        cache=False,
    )
    judge = LLMThoughtJudge(judge_lm, prompt_builder)

    with open(f"{judge_json_path}.console.log", "a", encoding="utf-8") as console_log:
        with contextlib.redirect_stdout(console_log), contextlib.redirect_stderr(
            console_log
        ):
            results = judge.judge_all(
                {"task": task.name, "case": case},
                candidates,
                max_candidates=args.max_judge_candidates,
            )
    write_judge_outputs(
        results,
        json_path=judge_json_path,
        csv_path=judge_csv_path,
        summary_path=judge_summary_path,
    )

    tokens = token_summary(full_json)
    return {
        "task": task.name,
        "case_id": case["id"],
        "model_name": args.model_name,
        "judge_model_name": args.judge_model_name or args.model_name,
        "entropy_field": args.entropy_field,
        "num_candidates": len(candidates),
        "num_judged_candidates": len(results),
        "full_graph_tokens": tokens,
        "paths": {
            "full_graph": full_graph_path,
            "judge_scores_json": judge_json_path,
            "judge_scores_csv": judge_csv_path,
            "judge_summary": judge_summary_path,
            "error_log": error_path,
        },
    }


def failed_case_summary(
    task: TaskSpec,
    case: Dict[str, Any],
    args: argparse.Namespace,
    run_root: str,
    error: Exception,
) -> Dict[str, Any]:
    case_dir = os.path.join(run_root, f"id{case['id']}")
    os.makedirs(case_dir, exist_ok=True)
    error_path = os.path.join(case_dir, "error.log")
    with open(error_path, "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    return {
        "task": task.name,
        "case_id": case["id"],
        "model_name": args.model_name,
        "judge_model_name": args.judge_model_name or args.model_name,
        "failed": True,
        "error": str(error),
        "paths": {"error_log": error_path},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full GoT and use an LLM judge to score whether each thought "
            "node is useful for the final answer."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--judge-model-name", default=None)
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
        default="doc_merge",
        help="Comma-separated task names, for example doc_merge,sorting_128.",
    )
    parser.add_argument("--data-ids", default=None)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--entropy-field", default="normalized_avg_entropy_bits")
    parser.add_argument("--max-judge-candidates", type=int, default=None)
    parser.add_argument("--include-root", action="store_true")
    parser.add_argument("--max-text-chars", type=int, default=2000)
    parser.add_argument("--max-predecessor-chars", type=int, default=800)
    parser.add_argument(
        "--full-graph-path",
        default=None,
        help="Optional existing full_graph.json path. Only valid for one task and one case.",
    )
    parser.add_argument(
        "--input-run-root",
        default=None,
        help=(
            "Optional folder containing existing id*/full_graph.json files, "
            "for example examples/batch_static_skip_results/doc_merge_skip0.2_TIME."
        ),
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of cases to judge in parallel. Default: 1.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "judge_analysis_results"),
    )
    args = parser.parse_args()

    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    if args.max_judge_candidates is not None and args.max_judge_candidates <= 0:
        raise ValueError("--max-judge-candidates must be positive")
    if args.parallel_workers <= 0:
        raise ValueError("--parallel-workers must be positive")
    if args.full_graph_path and args.input_run_root:
        raise ValueError("Use either --full-graph-path or --input-run-root, not both")

    task_specs = build_task_specs()
    tasks = selected_tasks(task_specs, args.tasks)
    data_ids = parse_ids(args.data_ids)
    if args.full_graph_path and (len(tasks) != 1 or data_ids is None or len(data_ids) != 1):
        raise ValueError("--full-graph-path requires exactly one task and one data id")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    max_cases = None if data_ids is not None else args.max_cases
    all_summaries: List[Dict[str, Any]] = []

    for task in tasks:
        run_root = os.path.join(args.output_dir, f"{task.name}_judge_{timestamp}")
        os.makedirs(run_root, exist_ok=True)
        cases = task.load_cases(data_ids, max_cases)
        task_summaries: List[Dict[str, Any]] = []
        if args.parallel_workers == 1:
            for case in cases:
                print(f"Running judge {task.name} id={case['id']}")
                try:
                    task_summaries.append(run_case_judge(task, case, args, run_root))
                except Exception as error:
                    task_summaries.append(
                        failed_case_summary(task, case, args, run_root, error)
                    )
        else:
            with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
                futures = {}
                for case in cases:
                    print(f"Running judge {task.name} id={case['id']}")
                    future = executor.submit(run_case_judge, task, case, args, run_root)
                    futures[future] = case
                for future in as_completed(futures):
                    case = futures[future]
                    try:
                        task_summaries.append(future.result())
                    except Exception as error:
                        task_summaries.append(
                            failed_case_summary(task, case, args, run_root, error)
                        )
        task_summaries.sort(key=lambda row: row.get("case_id", -1))

        task_summary_path = os.path.join(run_root, "judge_batch_summary.json")
        task_summary = {
            "task": task.name,
            "num_cases": len(task_summaries),
            "num_failed": sum(1 for row in task_summaries if row.get("failed")),
            "runs": task_summaries,
            "paths": {"run_root": run_root, "summary": task_summary_path},
        }
        with open(task_summary_path, "w", encoding="utf-8") as f:
            json.dump(task_summary, f, indent=2, ensure_ascii=False)
        all_summaries.append(task_summary)

    root_summary_path = os.path.join(
        args.output_dir,
        f"judge_analysis_summary_{timestamp}.json",
    )
    with open(root_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_tasks": len(all_summaries),
                "tasks": all_summaries,
                "paths": {"summary": root_summary_path},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
