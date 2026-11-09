# Applied Intelligence manuscript: shortened version

This package shortens the final two-author Springer Nature source prepared on
7 November 2026. The initial shortening used deletions only. The approved
10 November consistency pass makes targeted replacements in the abstract,
introduction, related work, Algorithm 1, and corpus-provenance paragraph;
removes the unused exact-match metric; and renumbers the appendices.
Author details, retained mathematics, table cells, funding, and availability
statements retain their original wording and values.

## Files

- `main.pdf` / `main.tex`: shortened main article.
- `supporting_appendices.pdf` / `supporting_appendices.tex`: supporting proofs,
  continuation protocol, native basis, structural compiler, locked-test protocol
  and technical amendment, and network profiles.
- `original_full.tex`: untouched full source, including the historical appendices
  removed from the submission version. It remains an archival source, not an
  additional file to upload as the main manuscript.
- `edit_manifest.json`: deletion record, exact targeted replacements, restored
  proof, source digest, and mechanical number maps.
- `shorten_manuscript.py`: deterministic editorial transformation; launches no
  experiments and recalculates no scientific results.
- `validation.json`: content-preservation and PDF build checks.

The main article removes the early pilot/48-run activation series and the
six-source development results. The 144-source continuation panel, all five
locked native comparison arms, the negative adaptive result, and the main
network comparison remain. Technical importer debugging, local output paths,
receipt hashes, and historical branch experiments remain in the full archive.

Supporting appendices run continuously from A to H. Their original letters
A, B, G, I, J, K, L, and N map to A, B, C, D, E, F, G, and H; old K.6
becomes F.5. All manuscript and supporting references follow this map.
The original J.4 format-bound argument now concludes E.3: a first-proof DAG
with at most two premise neighbours per derived vertex has degeneracy at most
two. Supporting tables and figures use S-prefixed numbers. Table S4 retains
the original positive differences, including +0.09375 at both horizons for
the whole-catalogue control. The supporting PDF has a descriptive subtitle.

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
