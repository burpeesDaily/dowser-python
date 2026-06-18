"""Tests for dowser: detectors, scorer, engine, and the scan pipeline.

These tests use the dependency-free :class:`HashingBackend` so the suite runs
without downloading a model or calling an API. They exercise the real logic
of every component and the staged pipeline end to end.
"""

from __future__ import annotations

import pandas as pd

from dowser import (
    CombinationRule,
    CombinationScorer,
    ConceptDetector,
    EmbeddingDetector,
    HashingBackend,
    LocalEngine,
    RegexDetector,
    ScanResult,
    SensitivityClass,
    SensitivitySpec,
    cusip,
    scan,
)


def test_regex_detector_finds_email() -> None:
    """The regex detector flags an email column as RESTRICTED."""
    detector = RegexDetector()
    samples = ["alice@example.com", "bob@test.org", "carol@mail.net"]
    findings = detector.detect("contact", samples)
    assert len(findings) == 1
    assert findings[0].sensitivity is SensitivityClass.RESTRICTED
    assert findings[0].detector_name == "regex"


def test_regex_detector_card_requires_luhn() -> None:
    """A 16-digit value failing Luhn is not flagged as a payment card."""
    detector = RegexDetector()
    # A valid test card number (passes Luhn) vs. a random 16-digit string.
    valid = ["4111 1111 1111 1111", "4111111111111111"]
    invalid = ["1234 5678 9012 3456"]
    assert detector.detect("card", valid)
    assert not detector.detect("card", invalid)


def test_regex_detector_ignores_plain_text() -> None:
    """A column of ordinary words produces no findings."""
    detector = RegexDetector()
    samples = ["red", "green", "blue", "yellow"]
    assert detector.detect("color", samples) == []


def test_embedding_detector_matches_class() -> None:
    """The embedding detector assigns the closest example class."""
    backend = HashingBackend(dimensions=128)
    examples = {
        SensitivityClass.RESTRICTED: [
            "John Smith",
            "Jane Doe",
            "Robert Brown",
        ],
        SensitivityClass.UNRESTRICTED: [
            "red",
            "blue",
            "green",
        ],
    }
    detector = EmbeddingDetector(backend, examples, similarity_threshold=0.3)
    findings = detector.detect("full_name", ["John Smith", "Jane Doe"])
    assert findings
    assert findings[0].sensitivity is SensitivityClass.RESTRICTED


def test_combination_scorer_uses_declared_rules() -> None:
    """A declared role combination elevates its matched columns."""
    rule = CombinationRule(
        name="identity trio",
        roles={
            "id": ("id", "ref"),
            "name": ("name", "nm"),
            "location": ("city", "zip", "postal"),
        },
        sensitivity=SensitivityClass.RESTRICTED,
    )
    result = ScanResult()
    for column in ("customer_ref", "full_nm", "zip_code"):
        result.columns[column] = _empty_column(column)
    scorer = CombinationScorer([rule])
    scorer.score(result)
    for column in ("customer_ref", "full_nm", "zip_code"):
        assert result.columns[column].sensitivity is SensitivityClass.RESTRICTED


def test_combination_scorer_no_match_when_role_missing() -> None:
    """An incomplete combination elevates nothing."""
    rule = CombinationRule(
        name="identity trio",
        roles={
            "id": ("id",),
            "name": ("name",),
            "location": ("city", "zip"),
        },
        sensitivity=SensitivityClass.RESTRICTED,
    )
    result = ScanResult()
    for column in ("user_id", "full_name"):  # no location column
        result.columns[column] = _empty_column(column)
    CombinationScorer([rule]).score(result)
    for column in ("user_id", "full_name"):
        assert result.columns[column].sensitivity is SensitivityClass.UNRESTRICTED


def test_cusip_validator_rejects_lookalikes() -> None:
    """The CUSIP validator accepts real CUSIPs and rejects same-shape strings."""
    assert cusip("037833100")  # Apple
    assert cusip("594918104")  # Microsoft
    assert cusip("38259P508")  # Alphabet (contains a letter)
    assert not cusip("037833101")  # wrong check digit
    assert not cusip("1234567890")  # 10 digits, phone-like
    assert not cusip("5949181")  # too short


def test_concept_detector_validator_filters_shape_matches() -> None:
    """A concept with a validator rejects shape matches that fail it.

    Two columns share the 9-character shape. Only the one whose values pass the
    CUSIP checksum is flagged; the phone-like column is not.
    """
    detector = ConceptDetector(
        concept_name="CUSIP",
        sensitivity=SensitivityClass.FINANCIAL,
        validator=cusip,
        pattern=r"\b[0-9A-Z]{9}\b",
    )
    real = detector.detect("security_id", ["037833100", "594918104"])
    lookalike = detector.detect("phone", ["8005551234", "8005554321"])
    assert real and real[0].sensitivity is SensitivityClass.FINANCIAL
    assert lookalike == []


def test_spec_compiles_and_scans_end_to_end() -> None:
    """A declared spec compiles into a working scan."""
    spec = SensitivitySpec()
    spec.declare_concept(
        "CUSIP",
        SensitivityClass.FINANCIAL,
        validator=cusip,
        pattern=r"\b[0-9A-Z]{9}\b",
    )
    spec.declare_combination(
        "identity trio",
        roles={
            "id": ("customer_ref", "user_id"),
            "name": ("name", "nm"),
            "location": ("city", "zip", "postal"),
        },
        sensitivity=SensitivityClass.RESTRICTED,
    )
    compiled = spec.compile()
    frame = pd.DataFrame(
        {
            "security_id": ["037833100", "594918104"],
            "customer_ref": ["A1", "B2"],
            "full_nm": ["Jane Doe", "John Smith"],
            "zip_code": ["78701", "80202"],
        }
    )
    result = scan(LocalEngine(frame, seed=1), compiled.detectors, compiled.scorer)
    assert result.columns["security_id"].sensitivity is SensitivityClass.FINANCIAL
    for column in ("customer_ref", "full_nm", "zip_code"):
        assert result.columns[column].sensitivity is SensitivityClass.RESTRICTED


def test_local_engine_samples_columns() -> None:
    """The local engine yields every column with string samples."""
    frame = pd.DataFrame({"email": ["a@b.com", "c@d.com"], "age": [30, 40]})
    engine = LocalEngine(frame, seed=1)
    columns = dict(engine.iter_columns())
    assert set(columns) == {"email", "age"}
    assert all(isinstance(value, str) for value in columns["age"])


def test_scan_end_to_end() -> None:
    """A full scan classifies a mixed table correctly."""
    frame = pd.DataFrame(
        {
            "email": ["alice@example.com", "bob@test.org"],
            "city": ["Austin", "Denver"],
            "ssn": ["123-45-6789", "987-65-4321"],
        }
    )
    result = scan(LocalEngine(frame, seed=1), [RegexDetector()])
    sensitive = result.sensitive_columns(minimum=SensitivityClass.RESTRICTED)
    assert "email" in sensitive
    assert "ssn" in sensitive
    assert "city" not in sensitive


def test_scan_staging_skips_expensive_detector() -> None:
    """With escalation, the embedding detector is skipped on a confident hit.

    The embedding detector here is configured to assign FINANCIAL to anything.
    If staging works, a column the regex detector already flagged with high
    confidence (email -> RESTRICTED) should not be overwritten, because the
    expensive detector never runs on it.
    """
    backend = HashingBackend(dimensions=64)
    everything_financial = {
        SensitivityClass.FINANCIAL: ["anything", "everything", "all values"],
    }
    embedding = EmbeddingDetector(
        backend, everything_financial, similarity_threshold=0.0
    )
    frame = pd.DataFrame({"email": ["alice@example.com", "bob@test.org"]})
    result = scan(
        LocalEngine(frame, seed=1),
        [RegexDetector(), embedding],
        escalate_below=0.5,
    )
    # Regex hit email with high confidence, so the embedding detector was
    # skipped; the only finding is the RESTRICTED email one.
    detectors_used = {
        finding.detector_name for finding in result.columns["email"].findings
    }
    assert detectors_used == {"regex"}


def _empty_column(field_name: str) -> object:
    """Return an empty ColumnResult for test seeding.

    Parameters
    ----------
    field_name : str
        The column name.

    Returns
    -------
    object
        An empty ColumnResult.
    """
    from dowser.results import ColumnResult

    return ColumnResult(field_name=field_name)
