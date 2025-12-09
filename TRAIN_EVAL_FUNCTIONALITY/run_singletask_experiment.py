"""
Entry-point for reproducing the SuTraN single-task baseline experiments.

The script exposes a minimal interface (log, task, seed) and dispatches
to the appropriate SST (sequential) or NSST (encoder-only) train/eval
pipeline with the exact hyperparameters reported in the paper.
"""

from __future__ import annotations

import argparse
from typing import Dict

from TRAIN_EVAL_FUNCTIONALITY import log_configs, technique_configs
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_NSST_SUTRAN import (
    train_eval as train_eval_nsst,
)
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_SST_SUTRAN import (
    train_eval as train_eval_sst,
)

_SST_TASKS = {"activity_suffix", "timestamp_suffix"}
_NSST_TASKS = {"remaining_runtime", "binary_outcome", "multiclass_outcome"}

# BPIC_17 variants only support multiclass outcomes; BPIC_19 has no outcome target.
_NO_OUTCOME_LOGS = {"BPIC_19"}
_MULTICLASS_ONLY_LOGS = {"BPIC_17", "BPIC_17_DR"}


def run_singletask_experiment(log_name: str, task: str, seed: int) -> None:
    """
    Train and evaluate the requested SuTraN single-task baseline.

    Parameters
    ----------
    log_name : {'BPIC_17', 'BPIC_17_DR', 'BPIC_19'}
        Identifier of the event log to use.
    task : {'remaining_runtime', 'binary_outcome', 'multiclass_outcome',
            'activity_suffix', 'timestamp_suffix'}
        Single-task objective to train.
    seed : int
        Random seed (1..5 in the paper experiments).
    """

    log_key = log_name.upper()
    if log_key not in log_configs.log_name_list:
        raise ValueError(
            f"'log_name' must be one of {log_configs.log_name_list}, got '{log_name}'."
        )

    task_key = task.lower()
    if task_key not in _SST_TASKS | _NSST_TASKS:
        raise ValueError(
            "Unsupported task '{task}'. "
            "Choose from 'remaining_runtime', 'binary_outcome', 'multiclass_outcome', "
            "'activity_suffix', or 'timestamp_suffix'."
        )

    median_caselen = log_configs.median_caselen_dict[log_key]

    if task_key in _SST_TASKS:
        train_eval_sst(
            log_name=log_key,
            median_caselen=median_caselen,
            seq_task=task_key,
            lr=0.0002,
            clen_dis_ref=True,
            seed=seed,
        )
        return

    # Below: NSST tasks.
    if task_key in {"binary_outcome", "multiclass_outcome"}:
        if log_key in _NO_OUTCOME_LOGS:
            raise ValueError(
                f"Outcome prediction is not available for '{log_key}'. "
                "Choose 'remaining_runtime', 'activity_suffix', or 'timestamp_suffix'."
            )
        if task_key == "binary_outcome" and log_key in _MULTICLASS_ONLY_LOGS:
            raise ValueError(
                f"'{log_key}' only features multiclass outcome prediction."
            )

    num_outclasses = log_configs.num_outclasses_dict[log_key]
    if task_key == "multiclass_outcome" and not num_outclasses:
        raise ValueError(
            f"No multiclass outcome labels available for '{log_key}'."
        )

    nsst_kwargs: Dict[str, object] = {
        "log_name": log_key,
        "median_caselen": median_caselen,
        "scalar_task": task_key,
        "out_mask": log_configs.out_masks_dict[log_key],
        "final_embedding": technique_configs.nsst_config_dict.get(
            "final_embedding", "last"
        ),
        "lr": 0.0002,
        "clen_dis_ref": True,
        "num_outclasses": num_outclasses if task_key == "multiclass_outcome" else None,
        "seed": seed,
    }

    train_eval_nsst(**nsst_kwargs)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the SuTraN single-task baselines."
    )
    parser.add_argument(
        "--log_name",
        required=True,
        choices=log_configs.log_name_list,
        help="Event log identifier.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(_SST_TASKS | _NSST_TASKS),
        help="Single-task objective to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed (1..5 in the paper experiments).",
    )
    return parser


if __name__ == "__main__":
    arguments = _build_arg_parser().parse_args()
    run_singletask_experiment(
        log_name=arguments.log_name,
        task=arguments.task,
        seed=arguments.seed,
    )
