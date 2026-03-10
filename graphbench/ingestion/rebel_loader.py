"""REBEL dataset loader for GraphBench knowledge graph construction.

Supports two ingestion paths:
- Pre-extracted path (local Mac dev): loads triples from a .parquet or .json file.
- REBEL inference path (Colab GPU): streams HotpotQA supporting-fact passages
  for downstream triple extraction via triple_extractor.extract_from_passages().

Entry point: stream_triples() — yields Triple dicts regardless of path taken.
Top-50 relation filter is applied in both paths.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

from graphbench.ingestion import Triple

logger = logging.getLogger(__name__)

# Top-50 REBEL relation types covering ~85% of extracted triples.
# Fixed research decision — not env-overridable.
TOP_50_RELATIONS: frozenset[str] = frozenset(
    {
        "country",
        "place_of_birth",
        "place_of_death",
        "occupation",
        "employer",
        "member_of",
        "part_of",
        "located_in_the_administrative_territorial_entity",
        "country_of_citizenship",
        "instance_of",
        "subclass_of",
        "follows",
        "followed_by",
        "spouse",
        "child",
        "father",
        "mother",
        "sibling",
        "cast_member",
        "director",
        "author",
        "publisher",
        "publication_date",
        "genre",
        "award_received",
        "educated_at",
        "field_of_work",
        "official_language",
        "capital",
        "head_of_government",
        "head_of_state",
        "contains_administrative_territorial_entity",
        "shares_border_with",
        "continent",
        "currency",
        "ethnic_group",
        "language_used",
        "religion",
        "notable_work",
        "developer",
        "operating_system",
        "programming_language",
        "licensed_to_broadcast_to",
        "record_label",
        "league",
        "sport",
        "position_played_on_team",
        "participant",
        "organizer",
        "presenter",
        "narrator",
    }
)


def _normalise_entity(text: str) -> str:
    """Lowercase and collapse whitespace in an entity surface form.

    Args:
        text: Raw entity string.

    Returns:
        Normalised entity string (may be empty if input was blank).
    """
    return " ".join(text.lower().split())


def stream_triples(
    *,
    preextracted_path: Path | None = None,
    max_triples: int = 60_000,
    split: str = "train",
) -> Iterator[Triple]:
    """Stream (subject, relation, object) triples for KG construction.

    Chooses between two paths based on whether a pre-extracted file exists:
    - Pre-extracted path: load from .parquet or .json (no GPU needed).
    - REBEL inference path: stream HotpotQA passages (requires GPU on Colab).
      In this path, the function yields passage dicts — the caller must run
      triple_extractor.extract_from_passages() and then call this again with
      the resulting .parquet saved to preextracted_path.

    Args:
        preextracted_path: Optional path to a pre-extracted triples file
            (.parquet with columns [subject, relation, object], or .json
            list of {subject, relation, object} dicts).
        max_triples: Maximum number of triples to yield after filtering.
        split: HotpotQA dataset split to use ("train" for maximum volume).

    Yields:
        Triple dicts with keys "subject", "relation", "object".
        All values are non-empty lowercase strings. Relation is snake_case.
    """
    if preextracted_path is not None and Path(preextracted_path).exists():
        logger.info("Loading pre-extracted triples from %s", preextracted_path)
        yield from _load_preextracted(Path(preextracted_path), max_triples)
    else:
        logger.warning(
            "No pre-extracted file at %s. Streaming HotpotQA passages for "
            "REBEL extraction. This path requires GPU — run on Colab. "
            "Use preextracted_path= to skip model inference on local Mac.",
            preextracted_path,
        )
        yield from _stream_hotpotqa_passages(split, max_triples)


def _load_preextracted(path: Path, max_triples: int) -> Iterator[Triple]:
    """Load triples from a pre-extracted parquet or JSON file.

    Args:
        path: Path to .parquet or .json file.
        max_triples: Maximum number of triples to yield.

    Yields:
        Filtered and normalised Triple dicts.

    Raises:
        ValueError: If the file lacks required columns.
        ValueError: If the file extension is not .parquet or .json.
    """
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Use .parquet or .json."
        )

    required_cols = {"subject", "relation", "object"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Pre-extracted file missing columns: {missing}")

    logger.info(
        "Loaded %d rows from %s. Filtering to TOP_50_RELATIONS...",
        len(df),
        path,
    )

    count = 0
    for _, row in df.iterrows():
        if count >= max_triples:
            break

        relation = str(row["relation"]).strip()
        if relation not in TOP_50_RELATIONS:
            continue

        subject = _normalise_entity(str(row["subject"]))
        object_ = _normalise_entity(str(row["object"]))

        if not subject or not object_:
            continue

        count += 1
        yield {"subject": subject, "relation": relation, "object": object_}

    logger.info("Yielded %d triples from pre-extracted file.", count)


def _stream_hotpotqa_passages(split: str, max_triples: int) -> Iterator[Triple]:
    """Stream HotpotQA supporting-fact passages as passage dicts.

    NOTE: This path does NOT yield Triple dicts — it yields passage dicts
    with keys {"passage_id", "text"} for downstream REBEL inference.
    After running triple_extractor.extract_from_passages() on these,
    save results to a .parquet file and use the pre-extracted path.

    Args:
        split: HotpotQA split to stream from.
        max_triples: Unused in this path (REBEL inference handles the cap).

    Yields:
        Passage dicts: {"passage_id": str, "text": str}.
    """
    # Import here to avoid loading datasets at module import time
    from datasets import load_dataset

    logger.info(
        "Streaming HotpotQA '%s' split for REBEL inference (GPU required).", split
    )

    dataset = load_dataset(
        "hotpot_qa", "distractor", split=split, streaming=True, trust_remote_code=True
    )

    seen_ids: set[str] = set()
    passage_count = 0

    for example in dataset:
        titles = example["context"]["title"]
        sentences_list = example["context"]["sentences"]

        for title, sentences in zip(titles, sentences_list):
            for sent_idx, sentence in enumerate(sentences):
                passage_id = f"{title}_{sent_idx}"
                if passage_id in seen_ids:
                    continue
                seen_ids.add(passage_id)
                text = sentence.strip()
                if text:
                    passage_count += 1
                    # Yield as a passage dict — caller runs REBEL on these
                    yield {"passage_id": passage_id, "text": text}  # type: ignore[misc]

    logger.info("Streamed %d unique HotpotQA passages.", passage_count)


def load_hotpotqa_passages(split: str = "train") -> Iterator[dict]:
    """Yield unique passage dicts from HotpotQA supporting facts.

    Flattens all context sentences from the HotpotQA dataset into unique
    (title, sentence_index) pairs. Deduplicates by passage_id to avoid
    redundant REBEL inference on repeated passages.

    Args:
        split: HotpotQA split to load from ("train", "validation").

    Yields:
        Dicts with keys:
            - "passage_id" (str): "{title}_{sentence_index}"
            - "text" (str): The sentence text.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "hotpot_qa", "distractor", split=split, streaming=True, trust_remote_code=True
    )

    seen_ids: set[str] = set()

    for example in dataset:
        titles = example["context"]["title"]
        sentences_list = example["context"]["sentences"]

        for title, sentences in zip(titles, sentences_list):
            for sent_idx, sentence in enumerate(sentences):
                passage_id = f"{title}_{sent_idx}"
                if passage_id in seen_ids:
                    continue
                seen_ids.add(passage_id)
                text = sentence.strip()
                if text:
                    yield {"passage_id": passage_id, "text": text}
