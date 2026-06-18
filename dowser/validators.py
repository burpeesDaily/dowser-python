"""SensitivitySpec: declare what is sensitive; let dowser build how to find it.

This module is the front door, and it is where dowser's premise lives. The
user does not write detectors. The user *declares* what is sensitive, in their
own terms, and the spec compiles those declarations into the detectors and
scorer that actually do the finding.

Declarations come in two kinds, mirroring the two ways sensitivity is defined
in the real world:

- A **concept declaration** names a kind of value that matters — ``"CUSIP"``,
  ``"MRN"``, an internal account format — and teaches it by example, and
  optionally by a validator (a check-digit function) and a shape pattern. The
  user says *what* the concept is; dowser figures out *how* to recognize it.

- A **combination declaration** names a set of field roles that are sensitive
  only together — id and name and location are PII in combination even when
  none is sensitive alone. The user declares the roles; dowser figures out
  which columns play them.

The :meth:`SensitivitySpec.compile` method turns a spec into the detectors and
scorer the scan engine consumes. That compile step is the whole point: it is
the boundary between the user's judgment (the spec) and dowser's mechanism
(the compiled detectors).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dowser.detectors import ConceptDetector, Detector, RegexDetector
from dowser.embeddings import EmbeddingBackend
from dowser.results import SensitivityClass
from dowser.scorers import CombinationRule, CombinationScorer
from dowser.validators import Validator


@dataclass(frozen=True, slots=True)
class ConceptDeclaration:
    """A user's declaration that a kind of value is sensitive.

    Attributes
    ----------
    name : str
        The concept's name, e.g. ``"CUSIP"``. Appears in findings.
    sensitivity : SensitivityClass
        How sensitive the concept is.
    examples : tuple[str, ...]
        Example values that teach dowser what the concept looks like. Used for
        embedding-based detection.
    validator : Validator | None
        Optional check function (e.g. a check-digit validator). A shape match
        that fails the validator is discarded — the primary defense against
        misidentification.
    pattern : str | None
        Optional regular expression describing the concept's shape. Cheap to
        evaluate and a useful first filter before the validator runs.
    """

    name: str
    sensitivity: SensitivityClass
    examples: tuple[str, ...] = ()
    validator: Validator | None = None
    pattern: str | None = None


@dataclass(frozen=True, slots=True)
class CombinationDeclaration:
    """A user's declaration that a set of field roles is sensitive together.

    Attributes
    ----------
    name : str
        The combination's name, e.g. ``"identity trio"``. Appears in findings.
    roles : dict[str, tuple[str, ...]]
        Maps each role name to keyword hints used to recognize which columns
        play that role. For example ``{"location": ("city", "zip", "postal")}``
        lets dowser match a ``zip_code`` column to the ``location`` role.
    sensitivity : SensitivityClass
        The sensitivity assigned to the fields when the full combination is
        present.
    """

    name: str
    roles: dict[str, tuple[str, ...]]
    sensitivity: SensitivityClass


@dataclass(slots=True)
class CompiledSpec:
    """The mechanism a spec compiles into: detectors plus a scorer.

    This is what the scan consumes. It is produced by
    :meth:`SensitivitySpec.compile`, not constructed directly.

    Attributes
    ----------
    detectors : list[Detector]
        The detectors to run, cheap first.
    scorer : CombinationScorer
        The scorer that applies declared combinations after detection.
    """

    detectors: list[Detector]
    scorer: CombinationScorer


class SensitivitySpec:
    """A declarative description of what an organization considers sensitive.

    Build a spec by declaring concepts and combinations, then call
    :meth:`compile` to turn it into the detectors and scorer a scan uses. The
    spec holds the user's judgment; the compiled output holds dowser's
    mechanism.

    Methods
    -------
    declare_concept(...)
        Declare that a kind of value is sensitive.
    declare_combination(...)
        Declare that a set of field roles is sensitive together.
    compile(backend=None, include_floor=True)
        Compile the declarations into a :class:`CompiledSpec`.

    Examples
    --------
    >>> from dowser import SensitivitySpec, SensitivityClass
    >>> from dowser.validators import cusip
    >>> spec = SensitivitySpec()
    >>> _ = spec.declare_concept(
    ...     "CUSIP",
    ...     SensitivityClass.FINANCIAL,
    ...     examples=["037833100", "594918104"],
    ...     validator=cusip,
    ...     pattern=r"\\b[0-9A-Z]{9}\\b",
    ... )
    >>> _ = spec.declare_combination(
    ...     "identity trio",
    ...     roles={
    ...         "id": ("id", "ref"),
    ...         "name": ("name", "nm"),
    ...         "location": ("city", "zip", "postal"),
    ...     },
    ...     sensitivity=SensitivityClass.RESTRICTED,
    ... )
    >>> compiled = spec.compile()
    >>> isinstance(compiled.detectors, list)
    True
    """

    def __init__(self) -> None:
        """Initialize an empty spec."""
        self._concepts: list[ConceptDeclaration] = []
        self._combinations: list[CombinationDeclaration] = []

    def declare_concept(
        self,
        name: str,
        sensitivity: SensitivityClass,
        examples: Sequence[str] | None = None,
        validator: Validator | None = None,
        pattern: str | None = None,
    ) -> "SensitivitySpec":
        """Declare that a kind of value is sensitive.

        Parameters
        ----------
        name : str
            The concept's name, e.g. ``"CUSIP"``.
        sensitivity : SensitivityClass
            How sensitive the concept is.
        examples : Sequence[str] | None, optional
            Example values that teach the concept, by default None.
        validator : Validator | None, optional
            A check function; shape matches that fail it are discarded. By
            default None.
        pattern : str | None, optional
            A regular expression for the concept's shape, by default None.

        Returns
        -------
        SensitivitySpec
            This spec, to allow chaining.
        """
        self._concepts.append(
            ConceptDeclaration(
                name=name,
                sensitivity=sensitivity,
                examples=tuple(examples or ()),
                validator=validator,
                pattern=pattern,
            )
        )
        return self

    def declare_combination(
        self,
        name: str,
        roles: dict[str, Sequence[str]],
        sensitivity: SensitivityClass,
    ) -> "SensitivitySpec":
        """Declare that a set of field roles is sensitive together.

        Parameters
        ----------
        name : str
            The combination's name.
        roles : dict[str, Sequence[str]]
            Maps each role name to keyword hints for matching columns.
        sensitivity : SensitivityClass
            Sensitivity assigned when the full combination is present.

        Returns
        -------
        SensitivitySpec
            This spec, to allow chaining.
        """
        self._combinations.append(
            CombinationDeclaration(
                name=name,
                roles={role: tuple(hints) for role, hints in roles.items()},
                sensitivity=sensitivity,
            )
        )
        return self

    def compile(
        self,
        backend: EmbeddingBackend | None = None,
        include_floor: bool = True,
    ) -> CompiledSpec:
        """Compile the declarations into detectors and a scorer.

        This is the boundary between judgment and mechanism. Each concept
        declaration becomes a :class:`~dowser.detectors.ConceptDetector`; the
        combination declarations become rules in a
        :class:`~dowser.scorers.CombinationScorer`. When ``include_floor`` is
        set, the universal :class:`~dowser.detectors.RegexDetector` is added
        first as the thin, honestly-incomplete baseline.

        Parameters
        ----------
        backend : EmbeddingBackend | None, optional
            The embedding backend for concepts taught by example. Concepts with
            examples are only given embedding detection when a backend is
            supplied. By default None.
        include_floor : bool, optional
            Whether to prepend the universal regex floor, by default True.

        Returns
        -------
        CompiledSpec
            The detectors and scorer the scan consumes.
        """
        detectors: list[Detector] = []
        if include_floor:
            detectors.append(RegexDetector())
        for concept in self._concepts:
            detectors.append(
                ConceptDetector(
                    concept_name=concept.name,
                    sensitivity=concept.sensitivity,
                    examples=concept.examples,
                    validator=concept.validator,
                    pattern=concept.pattern,
                    backend=backend,
                )
            )
        rules = [
            CombinationRule(
                name=combination.name,
                roles=combination.roles,
                sensitivity=combination.sensitivity,
            )
            for combination in self._combinations
        ]
        return CompiledSpec(detectors=detectors, scorer=CombinationScorer(rules))
