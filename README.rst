dowser-python
=============

**Sensitive data discovery that runs on a laptop or a lake.**

.. code-block:: bash

   pip install dowser-python

.. code-block:: python

   import dowser

A dowser finds hidden water. ``dowser`` finds hidden sensitive data — the PII,
PHI, financial, and confidential fields scattered across a data estate that
nobody fully mapped.

It is built on a premise most discovery tools get backwards. Existing tools
ship hundreds of predefined rules and, when you need one they don't have, make
you write the detector yourself — to tell the tool that CUSIP matters, you
have to encode *how* to find a CUSIP. That forces a domain expert to also be a
detection engineer.

``dowser`` inverts this. **You declare what is sensitive, in your own terms.
dowser figures out how to find it.**

This is the companion library to the article *Sensitive Data Discovery with
AI* in the `Demystifying Data Governance <https://www.formosa1544.com>`_ series.

The idea in one example
-----------------------

You declare a concept (taught by example, and optionally checked by a
validator) and a combination (roles that are sensitive together). You write no
detectors.

.. code-block:: python

   from dowser import SensitivitySpec, SensitivityClass, LocalEngine, scan
   from dowser.validators import cusip
   import pandas as pd

   spec = SensitivitySpec()

   # "CUSIP is sensitive." Taught by a shape and a check-digit validator.
   spec.declare_concept(
       "CUSIP",
       SensitivityClass.FINANCIAL,
       validator=cusip,
       pattern=r"\b[0-9A-Z]{9}\b",
   )

   # "An id, a name, and a location together are PII."
   spec.declare_combination(
       "identity trio",
       roles={
           "id": ("customer_ref", "user_id"),
           "name": ("name", "nm"),
           "location": ("city", "zip", "postal"),
       },
       sensitivity=SensitivityClass.RESTRICTED,
   )

   # dowser compiles declarations into the detectors and scorer.
   compiled = spec.compile()

   frame = pd.DataFrame({
       "security_id":   ["037833100", "594918104"],  # real CUSIPs
       "contact_phone": ["8005551234", "8005554321"], # same 9-ish shape
       "full_nm":       ["Jane Doe", "John Smith"],
       "zip_code":      ["78701", "80202"],
   })

   result = scan(LocalEngine(frame), compiled.detectors, compiled.scorer)
   print(result.sensitive_columns())

``security_id`` is flagged FINANCIAL — its values pass the CUSIP checksum.
``contact_phone`` is **not** flagged as a CUSIP — the phone numbers match the
shape but fail the checksum, so the validator rejects them. This is the
primary defense against misidentification: domain identifiers collide by
shape, and a check digit breaks the tie.

Why declarations, not rules
---------------------------

Sensitivity is contextual. A clinical code matters to a hospital, a CUSIP to a
bank, an in-game purchase to a game studio — none of them in any generic PII
dictionary. Aiming for a universal list of "sensitive things" is the wrong
goal. ``dowser`` starts from a thin, honestly-incomplete universal floor (email,
national ID, payment card, phone) and lets you teach the rest in your own
terms, so the tool ends up fitted to your domain rather than generically
adequate for everyone and ideal for no one.

How it works
------------

``dowser`` compiles a ``SensitivitySpec`` into small, replaceable parts:

- **Concept detectors** — one per declared concept. Each can use a shape
  pattern, a validator (a check function that rejects look-alikes), and
  examples compared by embedding similarity. The user supplies the *what*;
  dowser builds the *how*.
- **A combination scorer** — applies declared role combinations, figuring out
  which columns play each role.
- **Embedding backends** — the model layer behind example-based detection:
  local (``SentenceTransformerBackend``), hosted (``HostedEmbeddingBackend``), or
  the dependency-free ``HashingBackend`` for tests and demos.
- **Engines** — how each column is sampled. ``LocalEngine`` handles an in-memory
  DataFrame; the sampled-warehouse and Spark engines for lake scale are on the
  roadmap and implement the same ``ScanEngine`` protocol.

Install options
---------------

.. code-block:: bash

   pip install dowser-python              # core
   pip install dowser-python[pandas]      # + the local DataFrame engine
   pip install dowser-python[local]       # + local embeddings (sentence-transformers)
   pip install dowser-python[hosted]      # + hosted embeddings (OpenAI)

Roadmap
-------

``dowser`` is evolving. The laptop path is fully working today; the rest is being
built in the open:

- A **resolver** that handles shape-collisions with the full disambiguation
  hierarchy — validation, reference-set membership, context, and learned
  boundaries — and flags genuinely ambiguous fields for review rather than
  guessing.
- **Negative examples**, so the classifier learns the boundary between
  colliding concepts, not just the center of each.
- **Sampled and Spark engines** for lake-scale scanning.
- A **feedback loop** and evaluation harness, so corrections measurably improve
  the classifier over time and converge it on your domain.
- **Catalog and access-control adapters**, so tags change behavior rather than
  just describing it.

The code will keep changing. The premise behind it — declare what is sensitive,
let the tool figure out how to find it — is the durable part.

License
-------

MIT.
