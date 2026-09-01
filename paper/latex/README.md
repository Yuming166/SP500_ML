# ACL LaTeX draft

`main.tex` uses the official `acl-org/acl-style-files` template in anonymous
review mode. The checked `main.pdf` is the compiled draft. Copy `acl.sty` and
`acl_natbib.bst` from the official template into this directory (or build with
that directory on `TEXINPUTS` and `BSTINPUTS`) before compiling.

The verified 2026-09-02 build used:

- ACL style commit `d5adc823ff0f80f98c80405ca0ab66c68e684409`;
- `acl.sty` SHA-256 `19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de`;
- `acl_natbib.bst` SHA-256 `6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e`;
- Tectonic 0.17.0.

From this directory, a standard TeX Live build is:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The figures are generated from frozen results rather than edited by hand:

```bash
cd ../..
.venv/bin/python paper/scripts/generate_figures.py
```

The manuscript is configured as an anonymous long-paper draft. Before a real
submission, re-check the target ARR cycle's page limit, responsible NLP
checklist, anonymity requirements, and the exact official style-file revision.
