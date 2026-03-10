"""Checkpoint utilities for saving and loading training state.

Handles:
- Saving GNN model weights and optimizer state to CHECKPOINT_DIR.
- Loading the best checkpoint based on validation AUC-ROC.
- Google Drive integration for Colab persistence across sessions.
- Atomic writes (write to temp file, then rename) to prevent corruption.

The GNN must achieve test AUC-ROC > 0.75 before the benchmark phase begins.
Checkpoint filenames include epoch, val_auc, and timestamp for easy tracking.

Implementation: Phase 3 (GNN training).
"""
