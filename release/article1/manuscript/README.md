# Applied Intelligence manuscript: deletion-only cut

This package shortens the final two-author Springer Nature source prepared on
7 November 2026. Retained manuscript prose has not been paraphrased. Abstract,
author details, retained mathematics, table cells, funding, and availability
statements retain their original wording and values.

## Files

- `main.pdf` / `main.tex`: shortened main article.
- `supporting_appendices.pdf` / `supporting_appendices.tex`: supporting proofs,
  continuation protocol, native basis, structural compiler, locked-test protocol
  and technical amendment, and network profiles.
- `original_full.tex`: untouched full source, including the historical appendices
  removed from the submission version. It remains an archival source, not an
  additional file to upload as the main manuscript.
- `edit_manifest.json`: deletion record, source digest, and mechanical number maps.
- `shorten_manuscript.py`: deterministic editorial transformation; launches no
  experiments and recalculates no scientific results.
- `validation.json`: content-preservation and PDF build checks.

The main article removes the early pilot/48-run activation series and the
six-source development results. The 144-source continuation panel, all five
locked native comparison arms, the negative adaptive result, and the main
network comparison remain. Technical importer debugging, local output paths,
receipt hashes, and historical branch experiments remain in the full archive.

Supporting appendices preserve their original letters and subsection identifiers
(A, B, G.1, I.1–I.2, J.1–J.3, K.1–K.4/K.6, L, and N) so the manuscript's
references still identify the original supporting material. Supporting tables
and figures use S-prefixed numbers. Main headings and object references have
been mechanically renumbered after deletion; no prose rewriting accompanies
that adjustment. The supporting PDF has a descriptive subtitle.

For review, supply `main.pdf` as the manuscript and the supporting PDF as the
separate supporting document. This package does not submit anything to a journal,
grant a new code licence, or replace the complete experimental data archive.

## Rebuild

From this directory, using Python 3 and a LaTeX installation with `latexmk`:

```sh
python3 shorten_manuscript.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_appendices.tex
```

All image files, the class, and the bibliography style share the same folder.
References are embedded in each TeX source. `original_full.tex` uses the same
assets and can be compiled separately to inspect the unshortened version.
The original source and all 14 original image assets remain unchanged.
