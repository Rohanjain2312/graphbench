"""Centralised configuration for GraphBench.

All settings are loaded from environment variables (with defaults where safe).
Never import secrets directly — always go through this module.

Usage:
    from graphbench.utils.config import settings

    neo4j_uri = settings.neo4j_uri
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    Attributes:
        neo4j_uri: Neo4j AuraDB connection URI.
        neo4j_username: Neo4j username (default: "neo4j").
        neo4j_password: Neo4j password.
        hf_token: HuggingFace Hub access token.
        wandb_api_key: Weights & Biases API key.
        wandb_project: W&B project name (default: "graphbench").
        embedding_model: Sentence-transformer model name.
        llm_model: Instruction-tuned LLM model name.
        faiss_index_path: Path where FAISS index is stored/loaded.
        checkpoint_dir: Directory for saving training checkpoints.
        log_level: Python logging level string (default: "INFO").
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI",
    )
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(
        default="password",
        description="Neo4j password",
    )

    # HuggingFace
    hf_token: str | None = Field(default=None, description="HuggingFace Hub token")

    # Weights & Biases
    wandb_api_key: str | None = Field(default=None, description="W&B API key")
    wandb_project: str = Field(default="graphbench", description="W&B project name")

    # Models
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformer model for embeddings",
    )
    llm_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="LLM model name (HuggingFace or Ollama)",
    )

    # Paths
    faiss_index_path: Path = Field(
        default=Path("./data/faiss_index"),
        description="Path to store/load the FAISS index",
    )
    checkpoint_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Directory for model checkpoints",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Python logging level")

    # Benchmark constants (not env-overridable, kept here for single source of truth)
    embedding_dim: int = 384
    top_k_faiss: int = 10
    subgraph_hops: int = 2
    community_resolution: float = 0.8
    gnn_layers: int = 3
    gnn_heads: int = 4
    gnn_auc_threshold: float = 0.75


# Singleton — import this everywhere
settings = Settings()
