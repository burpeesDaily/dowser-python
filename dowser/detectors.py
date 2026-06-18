"""dowser: sensitive data discovery that runs on a laptop or a lake.

dowser inverts the usual division of labor in sensitive-data discovery. You
declare *what* is sensitive — in your own terms, by example — and dowser
figures out *how* to find it. Declarations live on a
:class:`~dowser.spec.SensitivitySpec`, which compiles into the detectors and
scorer a scan consumes. The library is the companion to the *Sensitive Data
Discovery with AI* article in the Demystifying Data Governance series.

Quick start
-----------
>>> import pandas as pd
>>> from dowser import SensitivitySpec, SensitivityClass, LocalEngine, scan
>>> from dowser.validators import cusip
>>> spec = SensitivitySpec()
>>> _ = spec.declare_concept(
...     "CUSIP",
...     SensitivityClass.FINANCIAL,
...     validator=cusip,
...     pattern=r"\\b[0-9A-Z]{9}\\b",
... )
>>> compiled = spec.compile()
>>> frame = pd.DataFrame({"security_id": ["037833100", "594918104"]})
>>> result = scan(LocalEngine(frame), compiled.detectors, compiled.scorer)
>>> result.sensitive_columns(SensitivityClass.FINANCIAL)
{'security_id': <SensitivityClass.FINANCIAL: 3>}
"""

from __future__ import annotations

from dowser.detectors import (
    ConceptDetector,
    Detector,
    EmbeddingDetector,
    RegexDetector,
)
from dowser.embeddings import (
    EmbeddingBackend,
    HashingBackend,
    HostedEmbeddingBackend,
    SentenceTransformerBackend,
)
from dowser.engines import LocalEngine, ScanEngine
from dowser.results import (
    ColumnResult,
    Finding,
    ScanResult,
    SensitivityClass,
)
from dowser.scan import scan
from dowser.scorers import CombinationRule, CombinationScorer
from dowser.spec import (
    CombinationDeclaration,
    CompiledSpec,
    ConceptDeclaration,
    SensitivitySpec,
)
from dowser.validators import Validator, cusip, luhn

__version__ = "0.1.0"

__all__ = [
    "scan",
    "SensitivitySpec",
    "ConceptDeclaration",
    "CombinationDeclaration",
    "CompiledSpec",
    "ScanEngine",
    "LocalEngine",
    "Detector",
    "RegexDetector",
    "EmbeddingDetector",
    "ConceptDetector",
    "EmbeddingBackend",
    "SentenceTransformerBackend",
    "HostedEmbeddingBackend",
    "HashingBackend",
    "CombinationScorer",
    "CombinationRule",
    "Finding",
    "ColumnResult",
    "ScanResult",
    "SensitivityClass",
    "Validator",
    "luhn",
    "cusip",
]
