"""Detectors: the components that examine a field and produce findings.

A detector looks at a column name and a sample of its values and returns zero
or more :class:`~dowser.results.Finding` objects. Detectors are the
extensible heart of dowser: the built-in :class:`RegexDetector` covers the
fast, well-defined patterns, the :class:`EmbeddingDetector` covers the cases
rules miss, and custom detectors plug in by implementing the
:class:`Detector` protocol.

The staged philosophy from the article lives here: cheap detectors
(:class:`RegexDetector`) run first and resolve most fields; expensive ones
(:class:`EmbeddingDetector`) run only where the cheap pass was inconclusive.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from dowser.embeddings import EmbeddingBackend
from dowser.results import Finding, SensitivityClass
from dowser.validators import Validator


@runtime_checkable
class Detector(Protocol):
    """Protocol every detector implements.

    A detector is any object with a ``name`` and a ``detect`` method. Using a
    :class:`typing.Protocol` means custom detectors do not need to inherit
    from a base class; they only need to match this shape.
    """

    name: str

    def detect(self, field_name: str, samples: Sequence[str]) -> list[Finding]:
        """Examine one field and return any findings.

        Parameters
        ----------
        field_name : str
            The column or field name.
        samples : Sequence[str]
            A sample of the field's values, as strings.

        Returns
        -------
        list[Finding]
            Zero or more findings for the field.
        """
        ...


# --- Built-in pattern library for the regex detector -----------------------
# Each entry maps a class name to a compiled pattern and the sensitivity it
# implies. Patterns are deliberately conservative; the embedding detector is
# what catches the cases these miss.


def _luhn_valid(digits: str) -> bool:
    """Return True if a digit string passes the Luhn checksum.

    Used to reduce false positives on credit-card detection: a 16-digit
    number that fails Luhn is almost certainly not a card.

    Parameters
    ----------
    digits : str
        A string of digits with no separators.

    Returns
    -------
    bool
        True if the Luhn check passes.
    """
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


class RegexDetector:
    """Fast, explainable detector for well-defined patterns.

    The regex detector is the cheap first pass. It recognizes the structured
    identifiers that have reliable shapes — email addresses, national
    identifiers, payment cards, phone numbers — using compiled patterns, and
    applies a Luhn check to candidate card numbers to suppress false hits.

    Attributes
    ----------
    name : str
        Detector name, used in findings and audit.

    Methods
    -------
    detect(field_name, samples)
        Return findings for any values matching a known pattern.
    """

    _EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    _SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
    _PHONE = re.compile(r"\b(?:\+?\d{1,2}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")

    def __init__(self, match_threshold: float = 0.3) -> None:
        """Initialize the detector.

        Parameters
        ----------
        match_threshold : float, optional
            Fraction of sampled values that must match a pattern before a
            finding is emitted, by default 0.3. This avoids flagging a whole
            column because a single value happened to look like an email.
        """
        self.name: str = "regex"
        self._match_threshold: float = match_threshold

    def detect(self, field_name: str, samples: Sequence[str]) -> list[Finding]:
        """Return findings for values matching a known pattern.

        Parameters
        ----------
        field_name : str
            The column name.
        samples : Sequence[str]
            Sampled values for the column.

        Returns
        -------
        list[Finding]
            One finding per pattern whose match rate clears the threshold.
        """
        if not samples:
            return []
        findings: list[Finding] = []
        checks = (
            ("email address", self._EMAIL, SensitivityClass.RESTRICTED, None),
            ("US SSN", self._SSN, SensitivityClass.RESTRICTED, None),
            ("payment card", self._CARD, SensitivityClass.FINANCIAL, _luhn_valid),
            ("phone number", self._PHONE, SensitivityClass.RESTRICTED, None),
        )
        for label, pattern, sensitivity, validator in checks:
            rate = self._match_rate(samples, pattern, validator)
            if rate >= self._match_threshold:
                findings.append(
                    Finding(
                        field_name=field_name,
                        sensitivity=sensitivity,
                        confidence=min(1.0, rate),
                        reason=f"{rate:.0%} of sampled values match {label}",
                        detector_name=self.name,
                    )
                )
        return findings

    def _match_rate(
        self,
        samples: Sequence[str],
        pattern: re.Pattern[str],
        validator: Callable[[str], bool] | None,
    ) -> float:
        """Return the fraction of samples matching a pattern.

        Parameters
        ----------
        samples : Sequence[str]
            Sampled values.
        pattern : re.Pattern[str]
            The compiled pattern to test.
        validator : Callable[[str], bool] | None
            Optional callable taking the matched digits and returning bool;
            used for the Luhn check on cards. ``None`` to skip validation.

        Returns
        -------
        float
            Matching fraction in [0.0, 1.0].
        """
        matches = 0
        for value in samples:
            found = pattern.search(value)
            if found is None:
                continue
            if validator is not None:
                digits = re.sub(r"\D", "", found.group())
                if not validator(digits):
                    continue
            matches += 1
        return matches / len(samples)


class EmbeddingDetector:
    """Similarity-based detector for the cases rules miss.

    The embedding detector represents both labelled example values and the
    column's sampled values as vectors, then assigns the column the class of
    whichever example set its samples most resemble. This catches sensitive
    data that has no fixed pattern — free-text names, addresses written a
    dozen ways, domain terms — which is exactly where the regex detector
    stops being useful.

    The embedding model itself is supplied through an
    :class:`~dowser.embeddings.EmbeddingBackend`, so the same detector runs
    on a local model (the default, private and free) or a hosted API
    (more capable, at a per-call cost) without any change to this class.

    Attributes
    ----------
    name : str
        Detector name, used in findings and audit.

    Methods
    -------
    detect(field_name, samples)
        Return a finding for the best-matching class above threshold.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        examples: dict[SensitivityClass, Sequence[str]],
        similarity_threshold: float = 0.6,
    ) -> None:
        """Initialize the detector and pre-embed the example sets.

        Parameters
        ----------
        backend : EmbeddingBackend
            The embedding model wrapper (local or hosted).
        examples : dict[SensitivityClass, Sequence[str]]
            Labelled example values per sensitivity class. These define what
            each class "looks like" in vector space.
        similarity_threshold : float, optional
            Minimum cosine similarity for a finding, by default 0.6.
        """
        self.name: str = "embedding"
        self._backend: EmbeddingBackend = backend
        self._threshold: float = similarity_threshold
        self._class_vectors: dict[SensitivityClass, list[list[float]]] = {
            sensitivity: backend.embed(list(values))
            for sensitivity, values in examples.items()
        }

    def detect(self, field_name: str, samples: Sequence[str]) -> list[Finding]:
        """Return a finding for the best-matching class above threshold.

        Parameters
        ----------
        field_name : str
            The column name.
        samples : Sequence[str]
            Sampled values for the column.

        Returns
        -------
        list[Finding]
            At most one finding, for the highest-scoring class that clears
            the similarity threshold.
        """
        if not samples:
            return []
        sample_vectors = self._backend.embed(list(samples))
        best_class: SensitivityClass | None = None
        best_score = 0.0
        for sensitivity, class_vectors in self._class_vectors.items():
            score = self._mean_max_similarity(sample_vectors, class_vectors)
            if score > best_score:
                best_score = score
                best_class = sensitivity
        if best_class is None or best_score < self._threshold:
            return []
        return [
            Finding(
                field_name=field_name,
                sensitivity=best_class,
                confidence=best_score,
                reason=(
                    f"values resemble {best_class} examples "
                    f"(similarity {best_score:.2f})"
                ),
                detector_name=self.name,
            )
        ]

    def _mean_max_similarity(
        self,
        sample_vectors: Sequence[Sequence[float]],
        class_vectors: Sequence[Sequence[float]],
    ) -> float:
        """Return the mean of each sample's best match to a class.

        For every sampled value, find its highest cosine similarity to any
        example in the class, then average those maxima. This rewards a
        column whose values each strongly resemble some example of the class.

        Parameters
        ----------
        sample_vectors : Sequence[Sequence[float]]
            Embedded sample values.
        class_vectors : Sequence[Sequence[float]]
            Embedded example values for one class.

        Returns
        -------
        float
            Mean best-match cosine similarity in [0.0, 1.0].
        """
        if not sample_vectors or not class_vectors:
            return 0.0
        total = 0.0
        for sample in sample_vectors:
            best = max(_cosine_similarity(sample, example) for example in class_vectors)
            total += best
        return total / len(sample_vectors)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity between two vectors.

    Parameters
    ----------
    left : Sequence[float]
        First vector.
    right : Sequence[float]
        Second vector.

    Returns
    -------
    float
        Cosine similarity in [-1.0, 1.0]; 0.0 if either vector is zero.
    """
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))


class ConceptDetector:
    """Detector for one user-declared concept, built from a declaration.

    A :class:`ConceptDetector` is what a concept declaration compiles into. It
    recognizes a single named concept — ``"CUSIP"``, an account format, a
    clinical code — using any combination of three signals the user supplied:

    1. A **shape pattern** (regex), the cheap first filter.
    2. A **validator** (e.g. a check-digit function) applied to every shape
       match; matches that fail are discarded. This is the primary defense
       against misidentification: a CUSIP-shaped string that fails the CUSIP
       checksum is not counted as a CUSIP.
    3. **Examples**, embedded and compared by similarity, for concepts that
       have no fixed shape or to corroborate a shape match.

    The user never writes this class directly. They declare the concept on a
    :class:`~dowser.spec.SensitivitySpec`, and the spec builds the detector.

    Attributes
    ----------
    name : str
        Detector name, ``"concept:<concept_name>"``, used in findings.

    Methods
    -------
    detect(field_name, samples)
        Return a finding for the concept if the signals support it.
    """

    def __init__(
        self,
        concept_name: str,
        sensitivity: SensitivityClass,
        examples: Sequence[str] = (),
        validator: Validator | None = None,
        pattern: str | None = None,
        backend: EmbeddingBackend | None = None,
        match_threshold: float = 0.3,
        similarity_threshold: float = 0.6,
    ) -> None:
        """Build a concept detector from a declaration's parts.

        Parameters
        ----------
        concept_name : str
            The concept's name, e.g. ``"CUSIP"``.
        sensitivity : SensitivityClass
            The class to assign when the concept is detected.
        examples : Sequence[str], optional
            Example values for embedding-based detection, by default ().
        validator : Validator | None, optional
            Check applied to each shape match; failures are discarded. By
            default None.
        pattern : str | None, optional
            Regular expression for the concept's shape, by default None.
        backend : EmbeddingBackend | None, optional
            Backend for embedding examples; without it, examples are unused.
            By default None.
        match_threshold : float, optional
            Fraction of sampled values that must match (and validate) the
            pattern before a finding is emitted, by default 0.3.
        similarity_threshold : float, optional
            Minimum mean similarity to the examples for an embedding finding,
            by default 0.6.
        """
        self.name: str = f"concept:{concept_name}"
        self._concept_name: str = concept_name
        self._sensitivity: SensitivityClass = sensitivity
        self._validator: Validator | None = validator
        self._pattern: re.Pattern[str] | None = (
            re.compile(pattern) if pattern is not None else None
        )
        self._match_threshold: float = match_threshold
        self._similarity_threshold: float = similarity_threshold
        self._backend: EmbeddingBackend | None = backend
        self._example_vectors: list[list[float]] = (
            backend.embed(list(examples)) if backend is not None and examples else []
        )

    def detect(self, field_name: str, samples: Sequence[str]) -> list[Finding]:
        """Return a finding for the concept if the signals support it.

        The pattern and validator are applied first (cheap, decisive). If they
        do not produce a finding and examples are available, similarity is
        tried. At most one finding is returned.

        Parameters
        ----------
        field_name : str
            The column name.
        samples : Sequence[str]
            Sampled values for the column.

        Returns
        -------
        list[Finding]
            At most one finding for this concept.
        """
        if not samples:
            return []
        pattern_finding = self._detect_by_pattern(field_name, samples)
        if pattern_finding is not None:
            return [pattern_finding]
        similarity_finding = self._detect_by_similarity(field_name, samples)
        if similarity_finding is not None:
            return [similarity_finding]
        return []

    def _detect_by_pattern(
        self, field_name: str, samples: Sequence[str]
    ) -> Finding | None:
        """Return a pattern-and-validator finding, or None.

        A value counts only if it matches the shape pattern and, when a
        validator is present, passes it. The validator is what rejects
        look-alikes of the same shape.

        Parameters
        ----------
        field_name : str
            The column name.
        samples : Sequence[str]
            Sampled values.

        Returns
        -------
        Finding | None
            A finding if the validated match rate clears the threshold.
        """
        if self._pattern is None:
            return None
        validated = 0
        for value in samples:
            match = self._pattern.search(value)
            if match is None:
                continue
            if self._validator is not None and not self._validator(match.group()):
                continue
            validated += 1
        rate = validated / len(samples)
        if rate < self._match_threshold:
            return None
        checked = " and validated" if self._validator is not None else ""
        return Finding(
            field_name=field_name,
            sensitivity=self._sensitivity,
            confidence=min(1.0, rate),
            reason=(
                f"{rate:.0%} of sampled values match{checked} "
                f"the {self._concept_name} pattern"
            ),
            detector_name=self.name,
        )

    def _detect_by_similarity(
        self, field_name: str, samples: Sequence[str]
    ) -> Finding | None:
        """Return an embedding-similarity finding, or None.

        Parameters
        ----------
        field_name : str
            The column name.
        samples : Sequence[str]
            Sampled values.

        Returns
        -------
        Finding | None
            A finding if mean similarity to the examples clears the threshold.
        """
        if self._backend is None or not self._example_vectors:
            return None
        sample_vectors = self._backend.embed(list(samples))
        if not sample_vectors:
            return None
        total = 0.0
        for sample in sample_vectors:
            best = max(
                _cosine_similarity(sample, example) for example in self._example_vectors
            )
            total += best
        score = total / len(sample_vectors)
        if score < self._similarity_threshold:
            return None
        return Finding(
            field_name=field_name,
            sensitivity=self._sensitivity,
            confidence=score,
            reason=(
                f"values resemble {self._concept_name} examples "
                f"(similarity {score:.2f})"
            ),
            detector_name=self.name,
        )
