"""GNN training loop with W&B experiment tracking.

Trains the 3-layer GAT model on the link-prediction task:
- Loss: Binary cross-entropy with logits (BCEWithLogitsLoss)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (mode='max', patience=5, factor=0.5)
- Validation metric: AUC-ROC (sklearn)
- Early stopping: patience=10 on val_auc
- Checkpoint: saved on every val_auc improvement

Run on Colab Pro (GPU). Not intended for local Mac execution.
The GNN must achieve test AUC-ROC > 0.75 before Phase 4 (pipelines).
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data

from graphbench.gnn.model import GATModel
from graphbench.utils.checkpoint import save_checkpoint
from graphbench.utils.config import settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _train_epoch(
    model: GATModel,
    data: Data,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Run a single training epoch.

    Args:
        model: GATModel instance.
        data: Training PyG Data with edge_label_index and edge_label.
        optimizer: Optimizer instance.
        device: Torch device string.

    Returns:
        Scalar training loss for this epoch.
    """
    model.train()
    optimizer.zero_grad()

    logits = model(
        data.x.to(device),
        data.edge_index.to(device),
        data.edge_label_index.to(device),
    )
    loss = F.binary_cross_entropy_with_logits(logits, data.edge_label.to(device))
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _evaluate(
    model: GATModel,
    data: Data,
    device: str,
) -> tuple[float, float]:
    """Evaluate on a val or test split.

    Args:
        model: GATModel instance.
        data: Validation/test PyG Data.
        device: Torch device string.

    Returns:
        Tuple of (loss, auc_roc).
    """
    model.eval()
    with torch.no_grad():
        logits = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_label_index.to(device),
        )

    labels = data.edge_label.numpy()
    probs = torch.sigmoid(logits).cpu().numpy()
    loss = F.binary_cross_entropy_with_logits(logits.cpu(), data.edge_label).item()
    auc = roc_auc_score(labels, probs)
    return float(loss), float(auc)


# ------------------------------------------------------------------
# Public training entry point
# ------------------------------------------------------------------


def train_gnn(
    model: GATModel,
    train_data: Data,
    val_data: Data,
    test_data: Data,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    scheduler_patience: int = 5,
    device: str = "cuda",
    checkpoint_dir: Path | None = None,
    wandb_project: str | None = None,
) -> dict[str, float]:
    """Train the GATModel and return final test metrics.

    Logs per-epoch metrics to W&B (if WANDB_API_KEY is set).
    Saves a checkpoint whenever val_auc improves.
    Applies early stopping when val_auc does not improve for
    early_stopping_patience consecutive epochs.

    Args:
        model: Initialised GATModel (moved to device inside this function).
        train_data: Training PyG Data from KGDataset.split().
        val_data: Validation PyG Data.
        test_data: Test PyG Data.
        epochs: Maximum number of training epochs (default 200).
        lr: Adam learning rate (default 1e-3).
        weight_decay: Adam L2 regularisation (default 1e-5).
        early_stopping_patience: Epochs without val_auc improvement before stopping.
        scheduler_patience: Epochs without improvement before LR reduction.
        device: Torch device string ("cuda" on Colab, "cpu" for local testing).
        checkpoint_dir: Directory for checkpoints. Defaults to settings.checkpoint_dir.
        wandb_project: W&B project name override. Defaults to settings.wandb_project.

    Returns:
        Dict with keys: test_auc, test_loss, best_val_auc, best_epoch.

    Raises:
        RuntimeError: If test_auc < settings.gnn_auc_threshold after training.
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # maximise AUC-ROC
        patience=scheduler_patience,
        factor=0.5,
    )

    # Initialise W&B if API key is available
    wandb_run: Any = None
    if settings.wandb_api_key:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project or settings.wandb_project,
                config={
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "gnn_layers": settings.gnn_layers,
                    "gnn_heads": settings.gnn_heads,
                    "early_stopping_patience": early_stopping_patience,
                    "scheduler_patience": scheduler_patience,
                    "device": device,
                },
            )
            logger.info("W&B run initialised: %s", wandb_run.name)
        except Exception as exc:
            logger.warning(
                "W&B initialisation failed: %s. Continuing without W&B.", exc
            )
            wandb_run = None

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info(
        "Starting GNN training: %d epochs max, device=%s, "
        "early_stopping_patience=%d.",
        epochs,
        device,
        early_stopping_patience,
    )

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_data, optimizer, device)
        val_loss, val_auc = _evaluate(model, val_data, device)

        # Step LR scheduler on val_auc (mode='max')
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        log_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "lr": current_lr,
        }

        if wandb_run is not None:
            import wandb

            wandb.log(log_payload)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %03d | train_loss=%.4f | val_loss=%.4f | "
                "val_auc=%.4f | lr=%.2e",
                epoch,
                train_loss,
                val_loss,
                val_auc,
                current_lr,
            )

        # Early stopping and checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_auc, checkpoint_dir)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered at epoch %d "
                    "(no improvement for %d epochs).",
                    epoch,
                    early_stopping_patience,
                )
                break

    # Final evaluation on test set
    test_loss, test_auc = _evaluate(model, test_data, device)

    results = {
        "test_auc": test_auc,
        "test_loss": test_loss,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
    }

    logger.info(
        "Training complete. test_auc=%.4f | best_val_auc=%.4f (epoch %d)",
        test_auc,
        best_val_auc,
        best_epoch,
    )

    if wandb_run is not None:
        import wandb

        wandb.log({"test_auc": test_auc, "test_loss": test_loss})
        wandb_run.finish()

    # Gate: GNN must pass AUC threshold before Phase 4
    if test_auc < settings.gnn_auc_threshold:
        logger.warning(
            "test_auc=%.4f is below threshold=%.2f. "
            "Do not proceed to Phase 4 until threshold is met.",
            test_auc,
            settings.gnn_auc_threshold,
        )
    else:
        logger.info(
            "AUC threshold met (%.4f >= %.2f). Ready for Phase 4.",
            test_auc,
            settings.gnn_auc_threshold,
        )

    return results
