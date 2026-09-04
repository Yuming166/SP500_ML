# Paper figures

Run the following command from the repository root:

```bash
.venv/bin/python paper/scripts/generate_figures.py
```

The script reads the frozen V11.1 and V12.1 summaries and formal records. It
writes vector PDF figures for LaTeX and 320-dpi PNG previews. The framework
figure is also exported as `framework_overview.svg` and uses original,
code-generated rounded cards and icons so it remains reproducible without
external artwork. The risk--coverage curve is descriptive except for the
preregistered 80% coverage operating point; the generated
`routing_curve_data.json` preserves that boundary explicitly.

The color palette is designed to remain distinguishable under common forms of
color-vision deficiency, and all plot text is at least 7.2 pt before scaling.
