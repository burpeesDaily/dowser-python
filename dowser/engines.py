"""Embedding backends: the pluggable model layer behind the AI detector.

The :class:`EmbeddingDetector` does not know or care where its vectors come
from. It talks to an :class:`EmbeddingBackend`, and any object matching that
protocol works. This is the seam that lets dowser run the same logic on a
local model (private, free, no network) or a hosted API (more capable, at a
per-call cost).

Three backends ship here:

- :class:`SentenceTransformerBackend` — the default. Runs a local model from
  the ``sentence-transformers`` package. No data leaves the machine.
- :class:`HostedEmbeddingBackend` — calls a hosted embeddings API. More
  capable for nuanced text, at the cost of a network dependency and per-call
  billing. The article's compounding-cost argument applies directly here.
- :class:`HashingBackend` — a deterministic, dependency-free fallback used
  for tests and quick demos. It is **not** semantically meaningful and must
  not be used in production.

The optional dependencies are imported lazily, inside the backends that need
them, so importing dowser never forces a heavy install.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for anything that turns text into vectors.

    A backend takes a list of strings and returns a list of equal-length
    float vectors, one per input, in the same order.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Parameters
        ----------
        texts : list[str]
            The texts to embed.

        Returns
        -------
        list[list[float]]
            One vector per input text, in input order.
        """
        ...


class SentenceTransformerBackend:
    """Local embedding backend using ``sentence-transformers``.

    This is the default backend: it downloads and runs a model locally, so no
    data ever leaves the host. The model is loaded once on first use.

    Attributes
    ----------
    model_name : str
        The sentence-transformers model identifier.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the backend without loading the model yet.

        The model is loaded lazily on the first :meth:`embed` call so that
        constructing the backend is cheap and import stays light.

        Parameters
        ----------
        model_name : str, optional
            The model to load, by default ``"all-MiniLM-L6-v2"`` — a small,
            fast, widely-used general-purpose model.
        """
        self.model_name: str = model_name
        self._model: "SentenceTransformer | None" = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with the local model, loading it on first use.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list[list[float]]
            One vector per input text.

        Raises
        ------
        ImportError
            If ``sentence-transformers`` is not installed.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError(
                    "SentenceTransformerBackend requires the "
                    "'sentence-transformers' package. Install it with "
                    "'pip install dowser[local]'."
                ) from error
            self._model = SentenceTransformer(self.model_name)
        vectors: Any = self._model.encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]


class HostedEmbeddingBackend:
    """Hosted embedding backend that calls an embeddings API.

    More capable than a small local model for nuanced text, at the cost of a
    network round trip and per-call billing. Because each call has a cost,
    the staged pipeline matters most when this backend is in use: run the
    cheap regex pass first, and only send the ambiguous remainder here.

    Attributes
    ----------
    model : str
        The hosted model identifier.
    """

    def __init__(
        self,
        client: Any,
        model: str = "text-embedding-3-small",
    ) -> None:
        """Initialize with an already-configured API client.

        The client is injected rather than constructed here, so dowser does
        not own credentials and the caller controls configuration.

        Parameters
        ----------
        client : Any
            A configured API client exposing an embeddings interface
            compatible with the OpenAI Python SDK.
        model : str, optional
            The embedding model, by default ``"text-embedding-3-small"``.
        """
        self.model: str = model
        self._client: Any = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts by calling the hosted API.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list[list[float]]
            One vector per input text, in input order.
        """
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class HashingBackend:
    """Deterministic, dependency-free backend for tests and demos.

    This backend hashes tokens into a fixed-width vector. It captures crude
    lexical overlap — texts sharing words land in similar directions — which
    is enough to exercise the detector logic and to demo the pipeline without
    downloading a model or calling an API. It is **not** semantically aware
    and must never be used in production.

    Attributes
    ----------
    dimensions : int
        The fixed vector width.
    """

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize the backend.

        Parameters
        ----------
        dimensions : int, optional
            Vector width, by default 64.
        """
        self.dimensions: int = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts by hashing their tokens into a bag-of-words vector.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list[list[float]]
            One L2-normalized vector per input text.
        """
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single text into a normalized hashed vector.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        list[float]
            An L2-normalized vector of length ``dimensions``.
        """
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
