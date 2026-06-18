"""The scan entry point: orchestrating detectors, engine, and scorer.

:func:`scan` is the one function most users call. It walks the columns an
engine yields, runs each detector over every column, collects the findings,
then lets the optional scorer add combination findings across columns. The
result is a :class:`~dowser.results.ScanResult` describing the sensitivity
of every column.

The staged philosophy from the article is expressed by the order and choice
of detectors the caller passes: put the cheap :class:`RegexDetector` first
and reserve the expensive :class:`EmbeddingDetector` for columns the cheap
pass left unresolved. :func:`scan` supports that directly through its
``escalate_below`` parameter.
"""

from __future__ import annotations

from collections.abc import Sequence

from dowser.detectors import Detector
from dowser.engines import ScanEngine
from dowser.results import ColumnResult, ScanResult, SensitivityClass  # noqa: F401
from dowser.scorers import CombinationScorer


def scan(
    engine: ScanEngine,
    detectors: Sequence[Detector],
    scorer: CombinationScorer | None = None,
    escalate_below: float | None = None,
) -> ScanResult:
    """Scan a dataset and return its sensitivity findings.

    Parameters
    ----------
    engine : ScanEngine
        The engine yielding columns and sampled values.
    detectors : Sequence[Detector]
        Detectors to run, in order. To stage cheap-before-expensive, list the
        regex detector before the embedding detector and set
        ``escalate_below``.
    scorer : CombinationScorer | None, optional
        A scorer to add combination findings after detection, by default
        None.
    escalate_below : float | None, optional
        Staging threshold, by default None. When set, detectors after the
        first run on a column only if the best confidence so far is below
        this value. This is how the expensive detector is skipped for columns
        the cheap one already resolved confidently.

    Returns
    -------
    ScanResult
        Findings for every column scanned.

    Examples
    --------
    >>> from dowser.engines import LocalEngine
    >>> from dowser.detectors import RegexDetector
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"email": ["a@b.com", "c@d.com"]})
    >>> result = scan(LocalEngine(frame), [RegexDetector()])
    >>> result.columns["email"].sensitivity
    <SensitivityClass.RESTRICTED: 4>
    """
    result = ScanResult()
    for field_name, samples in engine.iter_columns():
        best_confidence = 0.0
        for index, detector in enumerate(detectors):
            if (
                index > 0
                and escalate_below is not None
                and best_confidence >= escalate_below
            ):
                break
            for finding in detector.detect(field_name, samples):
                result.record(finding)
                if finding.confidence > best_confidence:
                    best_confidence = finding.confidence
        if field_name not in result.columns:
            # Ensure every scanned column appears, even with no findings.
            result.columns[field_name] = ColumnResult(field_name=field_name)
    if scorer is not None:
        scorer.score(result)
    return result
