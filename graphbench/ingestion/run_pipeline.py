"""Orchestrator for the Phase 2 ingestion pipeline.

Wires together: triple streaming → entity deduplication → embedding →
FAISS index building → Neo4j writing.

Usage (local Mac, pre-extracted triples):
    poetry run python -m graphbench.ingestion.run_pipeline \\
        --preextracted data/triples.parquet \\
        --dry-run

Usage (Colab, full REBEL inference):
    python -m graphbench.ingestion.run_pipeline --max-triples 55000
"""

import argparse
import logging
from pathlib import Path

from graphbench.ingestion import Triple
from graphbench.ingestion.embedder import embed_entities
from graphbench.ingestion.faiss_writer import build_and_save_index
from graphbench.ingestion.neo4j_writer import write_triples
from graphbench.ingestion.rebel_loader import stream_triples
from graphbench.utils.config import settings
from graphbench.utils.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    *,
    preextracted_path: Path | None = None,
    max_triples: int = 55_000,
    dry_run: bool = False,
) -> dict:
    """Orchestrate the full Phase 2 ingestion pipeline.

    Steps:
    1. Stream triples (from preextracted_path OR HotpotQA+REBEL).
    2. Deduplicate entities across all subject/object values.
    3. Embed unique entities with all-MiniLM-L6-v2.
    4. Build and save FAISS IndexFlatIP to disk.
    5. Write triples to Neo4j AuraDB (skipped if dry_run=True).

    Args:
        preextracted_path: If provided and file exists, skip REBEL inference.
        max_triples: Maximum number of triples to process.
        dry_run: If True, skip the Neo4j write step (useful for local testing).

    Returns:
        Summary dict: {"n_triples", "n_entities", "faiss_size"}.
    """
    logger.info("=== Phase 2: Ingestion Pipeline ===")
    logger.info(
        "max_triples=%d | dry_run=%s | preextracted=%s",
        max_triples,
        dry_run,
        preextracted_path,
    )

    # Step 1: Collect triples
    logger.info("Step 1/5 — Streaming triples...")
    triples: list[Triple] = list(
        stream_triples(preextracted_path=preextracted_path, max_triples=max_triples)
    )
    logger.info("Collected %d triples.", len(triples))

    if not triples:
        logger.error("No triples collected. Aborting pipeline.")
        return {"n_triples": 0, "n_entities": 0, "faiss_size": 0}

    # Step 2: Deduplicate entities
    logger.info("Step 2/5 — Deduplicating entities...")
    entity_set: set[str] = set()
    for t in triples:
        entity_set.add(t["subject"])
        entity_set.add(t["object"])
    entity_strings = sorted(entity_set)
    logger.info("Unique entities: %d", len(entity_strings))

    # Step 3: Embed
    logger.info("Step 3/5 — Embedding entities...")
    embeddings = embed_entities(entity_strings, show_progress=True)

    # Step 4: Build and save FAISS index
    logger.info("Step 4/5 — Building FAISS index...")
    build_and_save_index(
        entity_strings,
        embeddings,
        index_path=settings.faiss_index_path,
    )

    # Step 5: Write to Neo4j
    if dry_run:
        logger.info(
            "Step 5/5 — Skipping Neo4j write (dry_run=True). "
            "Would write %d triples.",
            len(triples),
        )
        n_written = 0
    else:
        logger.info("Step 5/5 — Writing triples to Neo4j AuraDB...")
        with Neo4jClient() as client:
            client.verify_connectivity()
            client.ensure_schema()
            n_written = write_triples(triples, client)
        logger.info("Neo4j write complete: %d triples.", n_written)

    summary = {
        "n_triples": len(triples),
        "n_entities": len(entity_strings),
        "faiss_size": len(entity_strings),
        "n_written_neo4j": n_written,
    }
    logger.info("=== Pipeline complete: %s ===", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphBench ingestion pipeline")
    parser.add_argument(
        "--preextracted",
        type=Path,
        default=None,
        help="Path to pre-extracted triples .parquet or .json",
    )
    parser.add_argument(
        "--max-triples",
        type=int,
        default=55_000,
        help="Maximum number of triples to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Neo4j write (test embedding/FAISS only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ingestion_pipeline(
        preextracted_path=args.preextracted,
        max_triples=args.max_triples,
        dry_run=args.dry_run,
    )
