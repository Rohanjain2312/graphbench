"""Triple extraction and normalisation from REBEL decoder output.

Parses the REBEL model's constrained decoding format into structured
(subject, relation, object) triples. Handles:
- Decoding the special <triplet>, <subj>, <obj> tokens.
- Deduplication of identical triples within a document.
- Entity mention normalisation (lowercasing, whitespace collapse).
- Filtering triples where subject or object is an empty string.

Also provides the REBEL model inference wrapper for Colab (GPU) runs.
"""

import logging
from collections.abc import Iterator

from graphbench.ingestion import Triple
from graphbench.ingestion.rebel_loader import TOP_50_RELATIONS

logger = logging.getLogger(__name__)


def parse_rebel_output(decoded_text: str) -> list[Triple]:
    """Parse REBEL constrained decoding output into structured triples.

    Handles the <triplet>/<subj>/<obj> token format produced by
    Babelscape/rebel-large. Deduplicates within a single decoded sequence.

    REBEL output format:
        "<triplet> SUBJECT <subj> OBJECT <obj> RELATION [<triplet> ...]"

    Args:
        decoded_text: Raw decoded string from the REBEL tokenizer with
            skip_special_tokens=False. Special tokens appear as literal
            strings in the decoded output.

    Returns:
        List of Triple dicts. Empty list if no valid triples found.
    """
    triplets: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()

    # Split on <triplet> token, skip the leading empty segment
    segments = decoded_text.split("<triplet>")[1:]

    for segment in segments:
        if "<subj>" not in segment or "<obj>" not in segment:
            continue

        parts = segment.split("<subj>")
        subject_raw = parts[0].strip()

        remainder = parts[1].split("<obj>")
        if len(remainder) < 2:
            continue

        object_raw = remainder[0].strip()
        relation_raw = remainder[1].strip()

        # Normalise: lowercase and collapse whitespace
        subject = " ".join(subject_raw.lower().split())
        object_ = " ".join(object_raw.lower().split())
        # Normalise relation: lowercase, spaces → underscores
        relation = "_".join(relation_raw.lower().split())

        if not subject or not object_ or not relation:
            continue

        key = (subject, relation, object_)
        if key not in seen:
            seen.add(key)
            triplets.append(
                {"subject": subject, "relation": relation, "object": object_}
            )

    return triplets


def filter_by_relation(
    triples: list[Triple],
    allowed: frozenset[str] = TOP_50_RELATIONS,
) -> list[Triple]:
    """Keep only triples whose relation is in the allowed set.

    Args:
        triples: List of Triple dicts.
        allowed: Set of allowed relation strings in snake_case.
            Defaults to TOP_50_RELATIONS from rebel_loader.

    Returns:
        Filtered list. Original list is not mutated.
    """
    return [t for t in triples if t["relation"] in allowed]


def extract_from_passages(
    passages: list[str],
    *,
    batch_size: int = 8,
    device: str = "cuda",
) -> Iterator[Triple]:
    """Run REBEL model inference on a list of text passages.

    Requires GPU. Loads Babelscape/rebel-large via HuggingFace Transformers.
    Intended for Colab Pro runs — not for local Mac development.

    Args:
        passages: List of text strings to extract triples from.
        batch_size: Number of passages per forward pass.
        device: Torch device string ("cuda" for Colab).

    Yields:
        Triple dicts filtered to TOP_50_RELATIONS.

    Raises:
        RuntimeError: If the REBEL model cannot be loaded (e.g., no GPU).
    """
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch are required for REBEL inference."
        ) from exc

    model_name = "Babelscape/rebel-large"
    logger.info("Loading REBEL model: %s on device=%s", model_name, device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load REBEL model '{model_name}'. "
            "Ensure you are running on Colab with GPU and HF_TOKEN is set."
        ) from exc

    logger.info(
        "REBEL model loaded. Processing %d passages in batches of %d.",
        len(passages),
        batch_size,
    )

    with torch.no_grad():
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            # forced_bos_token_id=0 is required for Babelscape/rebel-large
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=3,
                forced_bos_token_id=0,
            )
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=False)
                triples = parse_rebel_output(decoded)
                filtered = filter_by_relation(triples)
                logger.debug(
                    "Passage %d: extracted %d triples, %d after filter",
                    i,
                    len(triples),
                    len(filtered),
                )
                yield from filtered
