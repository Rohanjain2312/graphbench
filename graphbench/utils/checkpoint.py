"""Checkpoint utilities for saving and loading GNN training state.

Atomic writes (write to .tmp, then rename) prevent corruption on Colab
if the runtime is interrupted mid-save.

Filename convention:
    gat_epoch{epoch:04d}_auc{val_auc:.4f}_{YYYYMMDD_HHMMSS}.pt

load_best_checkpoint() scans CHECKPOINT_DIR and returns the checkpoint
with the highest val_auc parsed from the filename — no file loading needed
to find the best, only the winner is loaded into memory.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from graphbench.utils.config import settings

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"gat_epoch(\d{4})_auc(\d+\.\d+)_")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_auc: float,
    checkpoint_dir: Path | None = None,
) -> Path:
    """Save model and optimizer state atomically to CHECKPOINT_DIR.

    Writes to a .tmp file first, then renames to the final path. On POSIX
    systems (Linux/macOS), rename is atomic — a crash mid-write leaves the
    previous best checkpoint intact.

    Args:
        model: The GATModel instance to save.
        optimizer: The optimizer whose state to save.
        epoch: Current training epoch (used in filename).
        val_auc: Validation AUC-ROC at this checkpoint (used in filename).
        checkpoint_dir: Directory to save to. Defaults to settings.checkpoint_dir.

    Returns:
        Path to the saved .pt file.
    """
    resolved_dir = Path(checkpoint_dir or settings.checkpoint_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gat_epoch{epoch:04d}_auc{val_auc:.4f}_{timestamp}.pt"
    target_path = resolved_dir / filename
    tmp_path = target_path.with_suffix(".tmp")

    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "val_auc": val_auc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": timestamp,
    }

    torch.save(checkpoint, tmp_path)
    tmp_path.rename(target_path)
    logger.info("Checkpoint saved: %s (val_auc=%.4f)", target_path.name, val_auc)
    return target_path


def load_checkpoint(
    path: Path,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a specific checkpoint file.

    Args:
        path: Path to the .pt checkpoint file.
        map_location: torch.load map_location (default "cpu").

    Returns:
        Checkpoint dict with keys: epoch, val_auc, model_state_dict,
        optimizer_state_dict, timestamp.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    logger.info(
        "Loaded checkpoint: %s (epoch=%d, val_auc=%.4f)",
        path.name,
        checkpoint["epoch"],
        checkpoint["val_auc"],
    )
    return checkpoint


def load_best_checkpoint(
    checkpoint_dir: Path | None = None,
    *,
    map_location: str = "cpu",
) -> tuple[dict[str, Any], Path]:
    """Load the checkpoint with the highest val_auc from CHECKPOINT_DIR.

    Scans filenames to find the best without loading every file into memory.
    Only the winner is loaded via torch.load.

    Args:
        checkpoint_dir: Directory to scan. Defaults to settings.checkpoint_dir.
        map_location: torch.load map_location argument (default "cpu").

    Returns:
        Tuple of (checkpoint_dict, path_to_file).

    Raises:
        FileNotFoundError: If no valid checkpoints are found in checkpoint_dir.
    """
    resolved_dir = Path(checkpoint_dir or settings.checkpoint_dir)
    candidates = list(resolved_dir.glob("gat_epoch*.pt"))

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in: {resolved_dir}")

    def _auc_from_path(p: Path) -> float:
        m = _FILENAME_RE.search(p.name)
        return float(m.group(2)) if m else 0.0

    best_path = max(candidates, key=_auc_from_path)
    checkpoint = load_checkpoint(best_path, map_location=map_location)
    logger.info(
        "Best checkpoint: %s (val_auc=%.4f)", best_path.name, checkpoint["val_auc"]
    )
    return checkpoint, best_path


def list_checkpoints(checkpoint_dir: Path | None = None) -> list[dict[str, Any]]:
    """List all checkpoints in CHECKPOINT_DIR sorted by val_auc descending.

    Parses metadata from filenames without loading torch tensors.

    Args:
        checkpoint_dir: Directory to scan. Defaults to settings.checkpoint_dir.

    Returns:
        List of dicts with keys: path, epoch, val_auc. Sorted best-first.
    """
    resolved_dir = Path(checkpoint_dir or settings.checkpoint_dir)
    results = []
    for p in resolved_dir.glob("gat_epoch*.pt"):
        m = _FILENAME_RE.search(p.name)
        if m:
            results.append(
                {
                    "path": p,
                    "epoch": int(m.group(1)),
                    "val_auc": float(m.group(2)),
                }
            )
    return sorted(results, key=lambda d: d["val_auc"], reverse=True)
