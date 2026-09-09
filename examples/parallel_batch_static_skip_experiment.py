import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Sequence, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from examples.batch_static_skip_experiment import (
    build_task_specs,
    parse_ids,
    selected_tasks,
)


def chunk_items(items: Sequence[int], chunk_size: int) -> List[List[int]]:
    return [
        list(items[index : index + chunk_size])
        for index in range(0, len(items), chunk_size)
    ]


def split_items(items: Sequence[int], num_shards: int) -> List[List[int]]:
    shard_size = (len(items) + num_shards - 1) // num_shards
    return chunk_items(items, shard_size)


def ids_for_task(task, explicit_data_ids: str) -> List[int]:
    parsed_ids = parse_ids(explicit_data_ids)
    if parsed_ids is not None:
        return parsed_ids
    cases = task.load_cases(None, None)
    return [int(case["id"]) for case in cases]


def parse_paths(value: str) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _completed_case_from_path(
    case_dir: str,
    requested_skip_orders: Iterable[str],
) -> int:
    name = os.path.basename(case_dir)
    if not name.startswith("id") or not name[2:].isdigit():
        return -1

    required_paths = [os.path.join(case_dir, "full_graph.json")]
    required_paths.extend(
        os.path.join(case_dir, skip_order, "summary.json")
        for skip_order in requested_skip_orders
    )
    if all(os.path.exists(path) and os.path.getsize(path) > 0 for path in required_paths):
        return int(name[2:])
    return -1


def completed_cases_from_resume_roots(
    resume_roots: Sequence[str],
    task_name: str,
    requested_skip_orders: Iterable[str],
) -> Dict[int, str]:
    completed_cases: Dict[int, str] = {}
    for resume_root in resume_roots:
        if not os.path.isdir(resume_root):
            continue
        for current_dir, dirnames, _ in os.walk(resume_root):
            if not os.path.basename(current_dir).startswith("id"):
                continue
            if task_name not in current_dir:
                continue
            completed_case_id = _completed_case_from_path(
                current_dir,
                requested_skip_orders,
            )
            if completed_case_id >= 0 and completed_case_id not in completed_cases:
                completed_cases[completed_case_id] = current_dir
            dirnames[:] = []
    return completed_cases


def completed_ids_from_resume_roots(
    resume_roots: Sequence[str],
    task_name: str,
    requested_skip_orders: Iterable[str],
) -> Set[int]:
    return set(
        completed_cases_from_resume_roots(
            resume_roots,
            task_name,
            requested_skip_orders,
        )
    )


def copy_completed_cases(
    completed_cases: Dict[int, str],
    output_root: str,
) -> Dict[int, str]:
    copied_paths: Dict[int, str] = {}
    if not completed_cases:
        return copied_paths

    for case_id, source_dir in sorted(completed_cases.items()):
        target_dir = os.path.join(output_root, f"id{case_id}")
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        copied_paths[case_id] = target_dir
    return copied_paths


def write_resume_report(
    path: str,
    task_name: str,
    original_ids: Sequence[int],
    completed_ids: Set[int],
    remaining_ids: Sequence[int],
    resume_roots: Sequence[str],
    copied_paths: Dict[int, str],
) -> None:
    report: Dict[str, object] = {
        "task": task_name,
        "resume_roots": list(resume_roots),
        "num_original_ids": len(original_ids),
        "num_completed_ids": len(completed_ids),
        "num_remaining_ids": len(remaining_ids),
        "completed_ids": sorted(completed_ids),
        "remaining_ids": list(remaining_ids),
        "copied_completed_paths": {
            str(case_id): copied_paths[case_id]
            for case_id in sorted(copied_paths)
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch_static_skip_experiment.py in parallel data shards."
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
        default="doc_merge",
        help="Comma-separated task names. Default: doc_merge.",
    )
    parser.add_argument(
        "--data-ids",
        default=None,
        help="Comma-separated case ids. If omitted, all ids of each task are used.",
    )
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Cases per shard. If set, overrides --num-shards.",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--skip-ratio", type=float, default=0.2)
    parser.add_argument(
        "--skip-orders",
        nargs="+",
        choices=["low", "high"],
        default=["low", "high"],
    )
    parser.add_argument("--entropy-field", default="normalized_avg_entropy_bits")
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Comma-separated existing result roots. Completed id folders under these "
            "roots are skipped. A case is complete when full_graph.json and all "
            "requested skip-order summary.json files exist."
        ),
    )
    parser.add_argument(
        "--copy-resumed",
        action="store_true",
        help=(
            "Copy completed id folders found under --resume-from into the new "
            "run folder as direct id*/ folders."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(__file__),
            "parallel_batch_static_skip_results",
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch child processes.",
    )
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_size is not None and args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if args.max_workers is not None and args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_specs = build_task_specs()
    tasks = selected_tasks(task_specs, args.tasks)
    batch_script = os.path.join(os.path.dirname(__file__), "batch_static_skip_experiment.py")
    processes = []
    max_workers = args.max_workers or args.num_shards
    resume_roots = parse_paths(args.resume_from)

    for task in tasks:
        original_data_ids = ids_for_task(task, args.data_ids)
        data_ids = list(original_data_ids)
        completed_cases = completed_cases_from_resume_roots(
            resume_roots,
            task.name,
            args.skip_orders,
        )
        completed_cases = {
            case_id: path
            for case_id, path in completed_cases.items()
            if case_id in set(original_data_ids)
        }
        completed_ids = set(completed_cases)
        completed_ids = set(original_data_ids) & completed_ids
        if completed_ids:
            data_ids = [case_id for case_id in data_ids if case_id not in completed_ids]

        task_output_root = os.path.join(
            args.output_dir,
            f"{task.name}_parallel_skip{args.skip_ratio}_{timestamp}",
        )
        os.makedirs(task_output_root, exist_ok=True)
        copied_paths = (
            copy_completed_cases(completed_cases, task_output_root)
            if args.copy_resumed
            else {}
        )
        write_resume_report(
            os.path.join(task_output_root, "resume_report.json"),
            task.name,
            original_data_ids,
            completed_ids,
            data_ids,
            resume_roots,
            copied_paths,
        )

        if completed_ids:
            print(
                f"Resume {task.name}: skipped {len(completed_ids)} completed ids, "
                f"remaining {len(data_ids)} ids"
            )

        if args.shard_size is None:
            shards = split_items(data_ids, args.num_shards) if data_ids else []
        else:
            shards = chunk_items(data_ids, args.shard_size)
        shards = [shard for shard in shards if shard]
        if not shards:
            print(f"No remaining ids for {task.name}; nothing to launch.")
            continue

        for shard_index, shard_ids in enumerate(shards):
            while len(processes) >= max_workers:
                still_running = []
                for process, log_file, command in processes:
                    return_code = process.poll()
                    if return_code is None:
                        still_running.append((process, log_file, command))
                    else:
                        log_file.close()
                        if return_code != 0:
                            raise RuntimeError(
                                "Shard failed with exit code "
                                f"{return_code}: {' '.join(command)}"
                            )
                processes = still_running
                if len(processes) >= max_workers:
                    time.sleep(5)

            shard_log_path = os.path.join(
                task_output_root,
                f"launcher_shard{shard_index:02d}.log",
            )
            command = [
                args.python,
                batch_script,
                "--model-name",
                args.model_name,
                "--config-path",
                args.config_path,
                "--tasks",
                task.name,
                "--data-ids",
                ",".join(str(item) for item in shard_ids),
                "--skip-ratio",
                str(args.skip_ratio),
                "--skip-orders",
                *args.skip_orders,
                "--entropy-field",
                args.entropy_field,
                "--output-dir",
                task_output_root,
                "--direct-output-root",
                "--run-log-name",
                f"run_shard{shard_index:02d}.log",
                "--batch-summary-name",
                f"batch_summary_shard{shard_index:02d}.json",
            ]
            log_file = open(shard_log_path, "w", encoding="utf-8")
            print(
                f"Launching {task.name} shard{shard_index:02d} "
                f"ids={shard_ids[0]}-{shard_ids[-1]} count={len(shard_ids)}"
            )
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, log_file, command))

    failures = []
    for process, log_file, command in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failures.append((return_code, command))

    if failures:
        details = "\n".join(
            f"exit={return_code}: {' '.join(command)}"
            for return_code, command in failures
        )
        raise RuntimeError(f"{len(failures)} shard(s) failed:\n{details}")


if __name__ == "__main__":
    main()
