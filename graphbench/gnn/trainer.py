"""GNN training loop with W&B experiment tracking.

Trains the 3-layer GAT model on the link-prediction task using:
- Binary cross-entropy loss with logits
- Adam optimizer (lr=1e-3, weight_decay=1e-5)
- ReduceLROnPlateau scheduler (patience=5)
- Early stopping (patience=10, based on val AUC-ROC)

Logs per-epoch metrics to W&B: train_loss, val_loss, val_auc.
Saves best checkpoint to CHECKPOINT_DIR when val AUC improves.

Training must be run on Colab Pro (GPU). Not intended for local Mac.

Implementation: Phase 3 (GNN).
"""
