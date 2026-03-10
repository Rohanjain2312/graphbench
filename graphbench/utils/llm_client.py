"""LLM inference client for GraphBench.

Abstracts over two backends depending on environment:
- **Colab / GPU**: HuggingFace Transformers with Mistral-7B-Instruct-v0.2
  loaded in 4-bit quantization via bitsandbytes.
- **Local Mac (dev)**: Phi-3-mini via Ollama HTTP API (no GPU required).

Both backends expose a single ``generate(prompt)`` method that returns a
decoded answer string. The caller is responsible for formatting the prompt
using ``PROMPT_TEMPLATE`` from pipelines/base.py.

Backend selection:
- ``backend="auto"`` (default): tries Ollama first; falls back to HuggingFace.
- ``backend="ollama"``: always uses Ollama (raises if not reachable).
- ``backend="hf"``: always uses HuggingFace (requires GPU + HF_TOKEN).
"""

from __future__ import annotations

import logging
from typing import Literal

from graphbench.utils.config import settings

logger = logging.getLogger(__name__)

Backend = Literal["auto", "ollama", "hf"]

# Default Ollama model for local Mac dev
_OLLAMA_DEFAULT_MODEL = "phi3"
_OLLAMA_BASE_URL = "http://localhost:11434"


class LLMClient:
    """Unified LLM client supporting Ollama (local) and HuggingFace (Colab/GPU).

    Usage::

        # Auto-detect backend (Ollama preferred for local dev)
        client = LLMClient()
        answer = client.generate(prompt)

        # Force HuggingFace backend on Colab
        client = LLMClient(backend="hf", model="mistralai/Mistral-7B-Instruct-v0.2")
    """

    def __init__(
        self,
        model: str | None = None,
        backend: Backend = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> None:
        """Initialise the LLM client.

        Args:
            model: Model identifier. Defaults to ``settings.llm_model`` for
                HF backend, or ``phi3`` for Ollama backend.
            backend: One of ``"auto"``, ``"ollama"``, ``"hf"``.
                ``"auto"`` probes Ollama first and falls back to HF.
            max_new_tokens: Max tokens to generate per call.
            temperature: Sampling temperature (lower = more deterministic).
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._hf_pipeline = None  # lazy HF pipeline

        resolved_backend, resolved_model = self._resolve_backend(backend, model)
        self._backend: Backend = resolved_backend
        self._model: str = resolved_model

        logger.info(
            "LLMClient initialised: backend=%s, model=%s.", self._backend, self._model
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a response for a formatted prompt.

        Args:
            prompt: A fully-formatted prompt string (use build_prompt() in
                Pipeline to construct it from PROMPT_TEMPLATE).

        Returns:
            Generated answer string (stripped, no trailing whitespace).

        Raises:
            RuntimeError: If the selected backend is unreachable or fails.
        """
        if self._backend == "ollama":
            return self._generate_ollama(prompt)
        return self._generate_hf(prompt)

    @property
    def backend(self) -> str:
        """Active backend name."""
        return self._backend

    @property
    def model(self) -> str:
        """Active model identifier."""
        return self._model

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(
        self, requested: Backend, model: str | None
    ) -> tuple[str, str]:
        """Determine the active backend and model string.

        Returns:
            (backend_str, model_str) tuple.
        """
        if requested == "ollama":
            return "ollama", model or _OLLAMA_DEFAULT_MODEL

        if requested == "hf":
            return "hf", model or settings.llm_model

        # "auto": probe Ollama, fall back to HF
        if self._ollama_reachable():
            logger.info("Ollama reachable — using Ollama backend.")
            return "ollama", model or _OLLAMA_DEFAULT_MODEL

        logger.info("Ollama not reachable — falling back to HuggingFace backend.")
        return "hf", model or settings.llm_model

    @staticmethod
    def _ollama_reachable() -> bool:
        """Return True if the local Ollama server is responding."""
        try:
            import requests  # noqa: PLC0415

            resp = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Ollama backend
    # ------------------------------------------------------------------

    def _generate_ollama(self, prompt: str) -> str:
        """Call the Ollama HTTP API and return the generated text.

        Args:
            prompt: Formatted prompt string.

        Returns:
            Generated text from Ollama.

        Raises:
            RuntimeError: If the Ollama API call fails.
        """
        import requests  # noqa: PLC0415

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{_OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            text: str = resp.json().get("response", "").strip()
            return text
        except Exception as exc:
            raise RuntimeError(f"Ollama generate failed: {exc}") from exc

    # ------------------------------------------------------------------
    # HuggingFace backend
    # ------------------------------------------------------------------

    def _generate_hf(self, prompt: str) -> str:
        """Run the HuggingFace Transformers pipeline and return generated text.

        Loads the model once (4-bit quantized) and caches it in ``_hf_pipeline``.

        Args:
            prompt: Formatted prompt string.

        Returns:
            Generated text from HuggingFace model.

        Raises:
            RuntimeError: If model loading or generation fails.
        """
        if self._hf_pipeline is None:
            self._hf_pipeline = self._load_hf_pipeline()

        try:
            outputs = self._hf_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self._hf_pipeline.tokenizer.eos_token_id,
            )
            full_text: str = outputs[0]["generated_text"]
            # Strip the prompt prefix to return only the generated answer
            if full_text.startswith(prompt):
                full_text = full_text[len(prompt) :]
            return full_text.strip()
        except Exception as exc:
            raise RuntimeError(f"HuggingFace generate failed: {exc}") from exc

    def _load_hf_pipeline(self):  # type: ignore[return]
        """Load and return the HuggingFace text-generation pipeline.

        Uses 4-bit quantization (BitsAndBytesConfig) for GPU memory efficiency.
        Requires a CUDA-capable GPU.

        Returns:
            Loaded transformers.pipeline object.

        Raises:
            RuntimeError: If GPU is unavailable or model loading fails.
        """
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "HuggingFace backend requires a CUDA GPU. "
                    "Use backend='ollama' for local Mac dev."
                )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            login_token = settings.hf_token
            logger.info("Loading HuggingFace model: %s (4-bit).", self._model)

            tokenizer = AutoTokenizer.from_pretrained(
                self._model,
                token=login_token,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self._model,
                quantization_config=bnb_config,
                device_map="auto",
                token=login_token,
            )
            logger.info("HuggingFace model loaded successfully.")
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace backend requires transformers and bitsandbytes. "
                "Install them or use backend='ollama'."
            ) from exc
