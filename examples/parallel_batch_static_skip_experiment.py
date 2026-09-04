import argparse
import datetime
import os
import subprocess
import sys
import time
from typing import List, Sequence

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

    for task in tasks:
        data_ids = ids_for_task(task, args.data_ids)
        if args.shard_size is None:
            shards = split_items(data_ids, args.num_shards)
        else:
            shards = chunk_items(data_ids, args.shard_size)
        shards = [shard for shard in shards if shard]

        task_output_root = os.path.join(
            args.output_dir,
            f"{task.name}_parallel_skip{args.skip_ratio}_{timestamp}",
        )
        os.makedirs(task_output_root, exist_ok=True)

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

            shard_dir = os.path.join(task_output_root, f"shard{shard_index:02d}")
            os.makedirs(shard_dir, exist_ok=True)
            shard_log_path = os.path.join(shard_dir, "launcher.log")
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
                shard_dir,
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
