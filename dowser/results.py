"""Scan engines: the laptop-to-lake execution layer.

A scan engine answers one question for the rest of the library: *how do I get
a representative sample of each column's values?* The classifier logic does
not change between a 10,000-row CSV and a billion-row lake table — only the
way the samples are drawn does. That difference lives here, behind the
:class:`ScanEngine` protocol.

This release ships :class:`LocalEngine`, which samples an in-memory pandas
DataFrame and is the right tool for a single table that fits in memory. The
big-data engines named in the article — a sampled warehouse engine that
pushes a ``TABLESAMPLE`` query down to the source, and a Spark engine for the
lake — are on the roadmap and implement this same protocol when they land.
"""

import random

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ScanEngine(Protocol):
    """Protocol for anything that yields columns and sampled values.

    An engine iterates the columns of a dataset, yielding each column name
    alongside a sample of its values rendered as strings. The sampling
    strategy — full read, row sample, or pushed-down query — is the engine's
    concern, not the classifier's.
    """

    def iter_columns(self) -> Iterator[tuple[str, list[str]]]:
        """Yield each column name with a sample of its values.

        Yields
        ------
        tuple[str, list[str]]
            A column name and a list of sampled string values.
        """
        ...


class LocalEngine:
    """Scan engine for an in-memory pandas DataFrame.

    Suitable for a single table that fits in memory — a CSV, an extract, a
    development sample. For each column it draws a random sample of non-null
    values (capped at ``sample_size``) and renders them as strings.

    Attributes
    ----------
    sample_size : int
        Maximum number of values sampled per column.

    Methods
    -------
    iter_columns()
        Yield each column with its sampled values.
    """

    def __init__(
        self,
        frame: Any,
        sample_size: int = 200,
        seed: int | None = None,
    ) -> None:
        """Initialize the engine over a DataFrame.

        Parameters
        ----------
        frame : Any
            A pandas DataFrame to scan. Typed as ``Any`` so importing
            dowser does not require pandas at import time.
        sample_size : int, optional
            Maximum values sampled per column, by default 200. A few hundred
            values are plenty to classify a column and keep cost bounded.
        seed : int | None, optional
            Seed for reproducible sampling, by default None.
        """
        self.sample_size: int = sample_size
        self._frame: Any = frame
        self._random: random.Random = random.Random(seed)

    def iter_columns(self) -> Iterator[tuple[str, list[str]]]:
        """Yield each column name with a sample of its values.

        Null values are dropped before sampling so detectors see real
        content. When a column has more than ``sample_size`` non-null values,
        a random subset is drawn.

        Yields
        ------
        tuple[str, list[str]]
            A column name and its sampled string values.
        """
        for column in self._frame.columns:
            values = self._frame[column].dropna().tolist()
            sampled = self._sample(values)
            yield str(column), [str(value) for value in sampled]

    def _sample(self, values: Sequence[object]) -> list[object]:
        """Return a random sample of values, capped at ``sample_size``.

        Parameters
        ----------
        values : Sequence[object]
            The non-null values of a column.

        Returns
        -------
        list[object]
            All values when at or below the cap, else a random subset.
        """
        if len(values) <= self.sample_size:
            return list(values)
        return self._random.sample(list(values), self.sample_size)
