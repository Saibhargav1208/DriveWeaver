"""Load and summarize DriveWeaver experiment specifications.

The existing training scripts still use their Hydra config groups. This module
adds a lightweight top-level experiment spec for comparing the same ablations
across datasets and VLM feature sources.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf


REQUIRED_KEYS = (
    "name",
    "dataset_config",
    "ablation",
    "world_model",
    "planner",
)


def load_experiment(path: str | Path) -> DictConfig:
    """Load an experiment spec and validate required top-level keys."""
    spec_path = Path(path)
    cfg = OmegaConf.load(spec_path)

    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in {spec_path}: {missing}")

    return cfg


def iter_experiment_specs(root: str | Path = "configs/experiments") -> Iterable[Path]:
    """Yield experiment YAML files in stable name order."""
    root_path = Path(root)
    yield from sorted(root_path.glob("*.yaml"))


def summarize_experiment(cfg: DictConfig) -> str:
    """Return a compact text summary for CLI inspection."""
    lines = [
        f"experiment: {cfg.name}",
        f"ablation:   {cfg.ablation.name}",
        f"dataset:    {cfg.dataset_config}",
        f"vlm:        {cfg.get('vlm_config', None)}",
        f"world:      {cfg.world_model.get('checkpoint', None)}",
        f"future:     {cfg.world_model.get('future_slots_cache_dir', None)}",
        f"planner:    {cfg.planner.get('checkpoint', None)}",
        f"selection:  {cfg.planner.get('selection', None)}",
    ]

    status = cfg.get("status", None)
    if status is not None:
        lines.append(f"status:     {status}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DriveWeaver experiment specs")
    parser.add_argument("spec", nargs="?", help="Path to a YAML experiment spec")
    parser.add_argument("--list", action="store_true", help="List experiment specs")
    args = parser.parse_args()

    if args.list:
        for spec_path in iter_experiment_specs():
            print(spec_path)
        return

    if not args.spec:
        parser.error("provide a spec path or --list")

    cfg = load_experiment(args.spec)
    print(summarize_experiment(cfg))


if __name__ == "__main__":
    main()
