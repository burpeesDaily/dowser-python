"""Core domain model for dowser: sensitivity classes and scan results.

This module defines the vocabulary the rest of the library speaks in: the
sensitivity classes a field can carry, a single :class:`Finding` produced by
one detector, and the aggregated :class:`ColumnResult` and :class:`ScanResult`
containers.

The sensitivity hierarchy mirrors the one used throughout the Demystifying
Data Governance series: the more sensitive the class, the stricter the
downstream protection it implies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class SensitivityClass(IntEnum):
    """Sensitivity of a data field, ordered from least to most sensitive.

    Ordering matters: a higher value is more sensitive, so the combination
    scorer and the result aggregation can compare and take the maximum class
    seen for a field. Using :class:`enum.IntEnum` makes ``max`` and ordinary
    comparison operators work directly on members.

    Attributes
    ----------
    UNRESTRICTED : int
        Public-facing data. No protection required.
    INTERNAL : int
        Internal-only operational data. Low sensitivity.
    CONFIDENTIAL : int
        Trade secrets, internal plans. Protect in transit and at rest.
    FINANCIAL : int
        Bank accounts, cards, transactions. Strict protection.
    RESTRICTED : int
        PII and PHI. The highest tier; always protected and audited.
    """

    UNRESTRICTED = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    FINANCIAL = 3
    RESTRICTED = 4

    def __str__(self) -> str:
        """Return the lowercase member name for human-readable output."""
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class Finding:
    """A single sensitivity detection for one field, from one detector.

    A finding is immutable: detectors produce findings, and the scorer reads
    them. Nothing mutates a finding after it is created, which keeps the
    pipeline easy to reason about.

    Attributes
    ----------
    field_name : str
        The column or field the finding applies to.
    sensitivity : SensitivityClass
        The class this detector assigned to the field.
    confidence : float
        Confidence in the assignment, from 0.0 to 1.0.
    reason : str
        Human-readable justification. The reason is what turns the classifier
        from a black box into something a reviewer can assess.
    detector_name : str
        Name of the detector that produced the finding, for audit and for
        the feedback loop.
    """

    field_name: str
    sensitivity: SensitivityClass
    confidence: float
    reason: str
    detector_name: str

    def __post_init__(self) -> None:
        """Validate the confidence range.

        Raises
        ------
        ValueError
            If confidence is outside the closed interval [0.0, 1.0].
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


@dataclass(slots=True)
class ColumnResult:
    """All findings for a single column, plus the resolved top class.

    Attributes
    ----------
    field_name : str
        The column these findings describe.
    findings : list[Finding]
        Every finding produced for this column, across all detectors.
    """

    field_name: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def sensitivity(self) -> SensitivityClass:
        """Return the highest sensitivity class among the findings.

        A field is treated as sensitive as its most sensitive finding. When
        there are no findings, the field is unrestricted.

        Returns
        -------
        SensitivityClass
            The maximum class seen, or ``UNRESTRICTED`` if there are none.
        """
        if not self.findings:
            return SensitivityClass.UNRESTRICTED
        return max(finding.sensitivity for finding in self.findings)

    @property
    def confidence(self) -> float:
        """Return the confidence of the highest-sensitivity finding.

        Returns
        -------
        float
            Confidence of the top finding, or 1.0 when there are no findings
            (an unrestricted field is unrestricted with full confidence).
        """
        if not self.findings:
            return 1.0
        top = max(self.findings, key=lambda f: (f.sensitivity, f.confidence))
        return top.confidence

    def add(self, finding: Finding) -> None:
        """Append a finding to this column.

        Parameters
        ----------
        finding : Finding
            The finding to record.
        """
        self.findings.append(finding)


@dataclass(slots=True)
class ScanResult:
    """The full result of scanning a dataset: one entry per column.

    Attributes
    ----------
    columns : dict[str, ColumnResult]
        Maps each column name to its result.
    """

    columns: dict[str, ColumnResult] = field(default_factory=dict)

    def record(self, finding: Finding) -> None:
        """Record a finding, creating the column entry if needed.

        Parameters
        ----------
        finding : Finding
            The finding to record under its field name.
        """
        column = self.columns.get(finding.field_name)
        if column is None:
            column = ColumnResult(field_name=finding.field_name)
            self.columns[finding.field_name] = column
        column.add(finding)

    def sensitive_columns(
        self, minimum: SensitivityClass = SensitivityClass.CONFIDENTIAL
    ) -> dict[str, SensitivityClass]:
        """Return columns at or above a minimum sensitivity.

        Parameters
        ----------
        minimum : SensitivityClass, optional
            The lowest class to include, by default ``CONFIDENTIAL``.

        Returns
        -------
        dict[str, SensitivityClass]
            Column name to resolved class, for columns meeting the threshold.
        """
        return {
            name: result.sensitivity
            for name, result in self.columns.items()
            if result.sensitivity >= minimum
        }
