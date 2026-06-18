"""Combination scoring: catching sensitivity that only emerges in aggregate.

Some fields are not sensitive alone but identify a person in combination. An
identifier, a name, and a location together are PII even when none of them is
sensitive on its own. Detectors look at one field at a time and miss this. The
:class:`CombinationScorer` runs after detection, inspects which columns are
present, and elevates sensitivity when a declared combination of field roles
is all present.

Crucially, the combinations are **declared by the user**, not hardcoded here.
The user states the roles that matter — on a
:class:`~dowser.spec.SensitivitySpec` — and the scorer figures out which actual
columns play those roles, matching by the keyword hints the declaration
supplies. This keeps the judgment ("these roles together are sensitive") with
the user and the mechanism ("which column is the location field") with dowser.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dowser.results import Finding, ScanResult, SensitivityClass


@dataclass(frozen=True, slots=True)
class CombinationRule:
    """A declared combination of field roles that is sensitive together.

    Produced by compiling a
    :class:`~dowser.spec.CombinationDeclaration`; not usually built directly.

    Attributes
    ----------
    name : str
        The combination's name, used in findings.
    roles : dict[str, tuple[str, ...]]
        Maps each role to keyword hints used to match columns to the role.
    sensitivity : SensitivityClass
        Sensitivity assigned to the contributing fields when all roles match.
    """

    name: str
    roles: dict[str, tuple[str, ...]]
    sensitivity: SensitivityClass


class CombinationScorer:
    """Elevate sensitivity when a declared combination of roles is present.

    The scorer reads the columns in a :class:`ScanResult` and, for each
    declared :class:`CombinationRule`, checks whether every role can be matched
    to some column by the role's keyword hints. When a rule is fully satisfied,
    each contributing column gets a finding raising it to the rule's
    sensitivity. The scorer only adds findings; it never lowers a field's
    sensitivity.

    Attributes
    ----------
    name : str
        Scorer name, used in the findings it produces.

    Methods
    -------
    score(result)
        Inspect the result and add combination findings in place.
    """

    def __init__(self, rules: Sequence[CombinationRule]) -> None:
        """Initialize the scorer with declared combination rules.

        Parameters
        ----------
        rules : Sequence[CombinationRule]
            The combinations to apply, from the compiled spec.
        """
        self.name: str = "combination"
        self._rules: tuple[CombinationRule, ...] = tuple(rules)

    def score(self, result: ScanResult) -> None:
        """Add combination findings to a result, in place.

        Parameters
        ----------
        result : ScanResult
            The result to inspect and augment. Modified in place.
        """
        column_names = list(result.columns)
        for rule in self._rules:
            matched = self._match_roles(column_names, rule)
            if matched is None:
                continue
            for field_name in matched:
                result.record(
                    Finding(
                        field_name=field_name,
                        sensitivity=rule.sensitivity,
                        confidence=0.8,
                        reason=(
                            f"part of the declared '{rule.name}' combination, "
                            f"sensitive in aggregate"
                        ),
                        detector_name=self.name,
                    )
                )

    def _match_roles(
        self, column_names: Sequence[str], rule: CombinationRule
    ) -> list[str] | None:
        """Return the columns satisfying every role in the rule, or None.

        Each role is matched to the first column whose name contains one of the
        role's keyword hints. The rule is satisfied only when every role finds
        a distinct column.

        Parameters
        ----------
        column_names : Sequence[str]
            The columns present in the result.
        rule : CombinationRule
            The combination to match.

        Returns
        -------
        list[str] | None
            The matched columns if every role is covered, else None.
        """
        matched: list[str] = []
        used: set[str] = set()
        for hints in rule.roles.values():
            column = self._first_matching(column_names, hints, used)
            if column is None:
                return None
            matched.append(column)
            used.add(column)
        return matched

    def _first_matching(
        self,
        column_names: Sequence[str],
        hints: Sequence[str],
        used: set[str],
    ) -> str | None:
        """Return the first unused column matching any hint.

        Parameters
        ----------
        column_names : Sequence[str]
            Columns to search.
        hints : Sequence[str]
            Keyword hints for the role, matched case-insensitively as
            substrings of the column name.
        used : set[str]
            Columns already claimed by earlier roles, skipped here.

        Returns
        -------
        str | None
            The first unused matching column, or None.
        """
        for name in column_names:
            if name in used:
                continue
            lowered = name.lower()
            if any(hint.lower() in lowered for hint in hints):
                return name
        return None
