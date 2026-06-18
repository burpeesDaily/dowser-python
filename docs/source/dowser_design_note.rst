dowser — Design Note (v0.1)
===========================

This note records the design that the v0.1 code implements, and the boundary
between what ships now and what is staged as roadmap. It is the design
companion to the locked article blueprint. The code is the proof; this note
is the map.

----

The premise (what the design must embody)
-----------------------------------------

Existing SDD tools force the domain expert to also be a detection engineer: to
tell a tool that CUSIP matters, you must encode *how* to detect a CUSIP. The
division of labor is backwards.

dowser's design puts the boundary in the right place:

- **The user supplies judgment** — *what* is sensitive — as a declarative
  ``SensitivitySpec``.
- **dowser supplies mechanism** — *how* to find it — by compiling that spec
  into detectors and a scorer.

The ``compile()`` step is the architectural seam between the two. Everything in
the design serves that separation.

----

The declarative front door: ``SensitivitySpec``
------------------------------------------------

The user builds a spec with two kinds of declaration.

Concept declarations — "this kind of value is sensitive"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   spec.declare_concept(
       name="CUSIP",
       sensitivity=SensitivityClass.FINANCIAL,
       examples=["037833100", "594918104"],   # taught by example
       validator=cusip,                        # check-digit function
       pattern=r"\b[0-9A-Z]{9}\b",             # shape hint
   )

The user names the concept and supplies any of three signals: a shape pattern,
a validator, and examples. They never write the detector. ``compile()`` turns
each concept declaration into a ``ConceptDetector``.

Combination declarations — "these roles together are sensitive"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   spec.declare_combination(
       name="identity trio",
       roles={
           "id":       ("customer_ref", "user_id"),
           "name":     ("name", "nm"),
           "location": ("city", "zip", "postal"),
       },
       sensitivity=SensitivityClass.RESTRICTED,
   )

The user declares roles and keyword hints. dowser matches the roles to the
actual columns. ``compile()`` turns combination declarations into rules in a
``CombinationScorer``.

Compilation
~~~~~~~~~~~

.. code-block:: python

   compiled = spec.compile(backend=None, include_floor=True)
   # compiled.detectors : [RegexDetector (floor), ConceptDetector, ...]
   # compiled.scorer    : CombinationScorer(declared rules)
   result = scan(engine, compiled.detectors, compiled.scorer, escalate_below=0.5)

``include_floor=True`` prepends the thin universal ``RegexDetector`` (email,
national ID, payment card, phone) — the honestly-incomplete baseline so users
are not staring at a blank page. The floor is the *starting point*, never the
*product*.

----

Misidentification: the validator is the v0.1 defense
----------------------------------------------------

Domain identifiers collide by shape: a CUSIP (9 chars), a ZIP+4 (9 digits),
and a phone (10 digits) overlap. A detector matching shape alone mislabels.

The full disambiguation hierarchy (designed; see roadmap) is:

1. **Validation / check digit** — decisive where it exists. **Shipped in v0.1.**
2. **Reference-set membership** — decisive when a known list exists. Roadmap.
3. **Context** — column name, neighbors, table character. Roadmap.
4. **Negative examples / learned boundaries.** Roadmap.
5. **Flag as ambiguous** rather than guess. Roadmap (the resolver).

v0.1 implements signal 1 generally: a concept may carry a ``validator``, and
``ConceptDetector`` discards every shape match that fails it. Shipped validators:
``luhn`` (payment cards) and ``cusip`` (CUSIP check digit). Demonstrated: a phone
column matching the CUSIP shape is rejected because it fails the checksum.

----

What v0.1 ships
---------------

.. list-table::
   :header-rows: 1

   * - Component
     - Status
   * - ``SensitivitySpec`` + ``compile()`` (the declarative front door)
     - ✅ shipped
   * - ``ConceptDeclaration``, ``CombinationDeclaration``, ``CompiledSpec``
     - ✅ shipped
   * - ``ConceptDetector`` (pattern + validator + examples)
     - ✅ shipped
   * - ``RegexDetector`` (universal floor)
     - ✅ shipped
   * - ``EmbeddingDetector`` (example-driven, low-level API)
     - ✅ shipped
   * - Embedding backends: local / hosted / hashing
     - ✅ shipped
   * - Generalized validators; ``luhn`` + real ``cusip`` check digit
     - ✅ shipped
   * - ``CombinationScorer`` driven by **declared** rules
     - ✅ shipped
   * - ``LocalEngine`` (pandas), staged ``scan()`` with ``escalate_below``
     - ✅ shipped
   * - Typed result model (``Finding``, ``ColumnResult``, ``ScanResult``)
     - ✅ shipped
   * - Quality: mypy --strict clean, flake8 clean, black, 12 tests, doctests
     - ✅

What is staged as roadmap (designed, not built)
-----------------------------------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Why deferred
   * - ``Resolver`` — full collision hierarchy + "ambiguous" verdict
     - The article designs it; v0.1 ships signal 1 (validation) only
   * - Negative-example learning in the embedding path
     - Needs the feedback store to be useful
   * - ``SampledEngine`` (warehouse pushdown), ``SparkEngine`` (lake)
     - Laptop path proven first; lake path is the next build
   * - Feedback-loop persistence + eval harness
     - The mechanism that converges the tool on a domain
   * - Catalog / access-control / lineage / audit adapters
     - Integration layer; out of scope for a v0.1 core

This split keeps v0.1 honest: it fully demonstrates the **premise** (declare
what, compile to how) and the **misidentification defense** (check-digit
validation), with the resolver and lake engines as a clearly-marked,
already-designed roadmap.

----

Naming
------

- PyPI / install: ``dowser-python``
- Import: ``import dowser``

Matches the ``forest-python`` → ``import forest`` convention used elsewhere in the
author's work. (The bare ``dowser`` import name shares with an abandoned 2014
memory-profiler package; this is a known, low-risk collision.)
