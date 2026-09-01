# Pilot-LLM V12 preformal abort

Date: 2026-09-02 (Asia/Shanghai)

V12 stopped at its frozen auxiliary fail-fast gate before any validation-agent smoke or formal output existed.

- Frozen selection: 358 questions and 1,074 evidence sentences.
- Initial rewrite calls: 1,074.
- Frozen first repair calls: 3.
- Usable substitutes: 1,073.
- Unusable substitutes: 1.
- Failed evidence ID: `boolq-1592052e5f54e039-e03`.
- Failure mode: `repair_overlong`.
- Source length window: 7 through 19 whitespace tokens.
- Initial candidate length: 20 tokens.
- First repair candidate length: 20 tokens.
- Substitute manifest SHA-256: `779ad8caee82b029cff504fccf55bca10735f4c4bb7f38b838ff7bdf753f4ffc`.
- Substitute statistics SHA-256: `d8db675f2eb80ddc941048ebb8bc731f1e0bd48fc8a95bea6d3f6ad85b907c9f`.

The frozen V12 rule prohibited truncation and required abort if any rewrite remained unusable. No question was dropped, no outcome was inspected, and no V12 validation-agent call was made. Any repair is a separately preregistered protocol amendment.
