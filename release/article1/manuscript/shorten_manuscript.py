#!/usr/bin/env python3
"""Reproduce a deletion-only editorial cut, with mechanical reference renumbering.

No scientific run, model fitting, or result recalculation occurs here.
The untouched input is original_full.tex beside this script.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / "original_full.tex").read_text()
changes = []


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def between(text, start, end):
    return text[text.index(start):text.index(end)]


def delete(text, fragment, reason, document="main"):
    assert text.count(fragment) == 1, (reason, fragment[:100], text.count(fragment))
    changes.append({"document": document, "reason": reason,
                    "removed_characters": len(fragment), "removed_sha256": sha(fragment),
                    "original_start_line": SOURCE[:SOURCE.index(fragment)].count("\n") + 1
                    if fragment in SOURCE else None})
    return text.replace(fragment, "", 1)


def cut(text, start, end, reason, document="main"):
    return delete(text, between(text, start, end), reason, document)


def drop_paragraph(text, start, reason, document="main"):
    pos = text.index(start)
    end = text.index("\n\n", pos)
    return delete(text, text[pos:end + 2], reason, document)


def is_subsequence(short, full):
    cursor = iter(full)
    return all(any(x == char for x in cursor) for char in short)


APP_A = r"\section*{Appendix A."
DECL = r"\section*{Statements and Declarations}"
BIB = r"\begin{thebibliography}"
main = cut(SOURCE, APP_A, DECL, "Separate all appendices from the main manuscript")
main = drop_paragraph(main, "The rest of the paper follows this structure.", "Remove roadmap referring to the complete historical appendix set")
main = cut(main, "Ideally, we would rapidly generate", r"\textbf{The inverse problem}", "Remove longer-term programme detail from the introduction")
main = delete(main, "Earlier checks measure how much of an exact finite benchmark the generator covers and how it continues from existing anchors; the skeleton's full return across a sequence of subsequent tasks remains a separate question.", "Remove reference to historical benchmark series")
main = delete(main, "Earlier tests of contextual transfer between branches motivate these distinctions, but the basic guarantees rely on standard proof-system properties: we claim no novelty when we infer forwards, combine strategies, or add provable lemmas.", "Remove historical branch discussion")
main = drop_paragraph(main, r"Urban \citep{Urban2013BliStr}", "Remove prover-cooperation paragraph tied to discarded branch studies")
main = drop_paragraph(main, "Earlier experiments in Appendices D--E", "Remove historical coverage protocol")
main = drop_paragraph(main, r"The operator \(\mathsf{S}", "Remove future inverse-task formalisation")
main = delete(main, "A frontier of sample means describes the observed trade-offs, whereas paired outcomes show whether individual schedules share them. Neither frontier establishes a population optimum, and changing the continuation horizon changes the quality coordinate rather than merely extending the same cost axis.", "Remove Pareto caveats repeated with the results")
main = cut(main, "The historical generational implementation records", "Only the initial axioms receive permanent protection", "Remove implementation-storage chronology and serialisation accounting")
main = drop_paragraph(main, r"Transfer tests freeze \(D_{e}\)", "Remove protocol of the discarded fragment-activation series")
main = cut(main, r"\subsection*{5.3 Library storage", r"\subsection*{5.4 Continuous construction", "Remove active/passive treatments of the historical series")
main = delete(main, "Historical generational panels commit complete layers, whereas native continuations charge individual primitive attempts and exclude any result that completes after a horizon. Each protocol fixes its own accounting boundary before the run.", "Remove historical accounting comparison")
main = delete(main, "The separate K--S--Peirce (KSP) oracle in Appendix G uses different axioms and counts leaves instead. We therefore keep its counts separate from the main series.", "Keep separate identity example in supporting proofs only")
main = cut(main, r"\subsection*{6.2 Protocol for the library", r"\subsection*{6.3 Native development", "Remove pilot and historical 48-run methods and Table 1")
main = cut(main, r"\subsection*{7.1 Longer runs", r"\subsection*{7.3 Neutral continuation", "Remove historical activation results, Table 2, and Figure 1")
main = cut(main, r"\subsection*{7.4 Structural enrichment", r"\subsection*{7.5 Locked native", "Remove six-source development results, Table 4, and Figure 3")
main = delete(main, "Appendix J preserves the earlier native study, whereas ", "Remove chronology clause preceding current network appendix")
main = cut(main, "Bounded activity improves formula output", "The coding library stores exact closed first proofs", "Remove discussion of the discarded historical 48-run series")
main = delete(main, "The historical 108-statement and native 919-statement references already informed development; they do not supply independent confirmation.", "Remove discarded human-coverage history from discussion")
main = cut(main, "The inverse problem takes a particular statement", "\n\nWe develop a cumulative method", "Remove repeated inverse/control programme from conclusions")
main = cut(main, "This checked backbone supplies the first stage", "\n\n\\section*{10 Reproducibility}", "Remove repeated future programme from conclusion")
main = delete(main, "Appendices A--N retain proofs, historical controls, benchmark provenance, and network records. The editorial manifest checks input hashes and identifies the first-source images.", "Remove broad historical-archive claim from main reproducibility section")
main = drop_paragraph(main, r"\textbf{Computing resources.}", "Remove historical execution-environment paragraph and obsolete cross-reference")
assert is_subsequence(main, SOURCE), "Main text must contain deletions only before numbering"
main_before_numbering = main

# Supporting material retains original appendix identifiers, so references in
# the manuscript still lead to the same proofs and protocol sections.
header = SOURCE[:SOURCE.index(r"\section*{1. Introduction}")]
abstract = between(header, r"\abstract{", r"\keywords{")
header = header.replace(abstract, "")
header = re.sub(r"\\keywords\{[^\n]*\}\n", "", header)
header = header.replace(r"\subtitle{Epochs limit active premises while certificates preserve proofs}",
                        r"\subtitle{Supporting appendices}")
parts = [between(SOURCE, APP_A, r"\section*{Appendix C."),
         between(SOURCE, r"\section*{Appendix G.", r"\subsection*{G.2."),
         between(SOURCE, r"\section*{Appendix I.", r"\subsection*{I.3."),
         between(SOURCE, r"\section*{Appendix J.", r"\subsection*{J.4."),
         between(SOURCE, r"\section*{Appendix K.", r"\subsection*{K.5."),
         between(SOURCE, r"\subsection*{K.6.", r"\section*{Appendix L."),
         between(SOURCE, r"\section*{Appendix L.", r"\section*{Appendix M."),
         between(SOURCE, r"\section*{Appendix N.", DECL)]
support_body = "".join(parts)
assert is_subsequence(support_body, SOURCE)
support = header + support_body + SOURCE[SOURCE.index(BIB):]

# Delete operational identifiers, implementation debugging, and redundant old
# network diagnostics; keep the actual technical amendment and negative result.
support = cut(support, "Frozen protocol SHA-256:", r"\subsection*{I.2.", "Remove receipt hashes", "support")
support = cut(support, "The following details accompany the main table (Table 33).", r"\section*{Appendix K.", "Remove superseded network table for the inherited core", "support")
support = drop_paragraph(support, "Package: ", "Remove local package path", "support")
support = cut(support, "Seal SHA-256:", "Input checks exclude", "Remove original seal hash", "support")
support = cut(support, "The author-supplied POI archive carries SHA-256", "Its MD5 agrees", "Remove archive checksum", "support")
support = delete(support, "Its MD5 agrees with the checksum on the author's artefact page. ", "Remove checksum commentary", "support")
support = cut(support, "The original v2 pipeline stopped", r"\subsection*{L.6.", "Remove importer failure trace, diagnostic hashes, and file list", "support")
support = drop_paragraph(support, "Read-only checks of all five arms", "Remove repair smoke-test chronology", "support")
support = cut(support, "The replacement package,", "It retains the same unopened", "Remove replacement path and seal hash", "support")
support = cut(support, "Evaluation receipt SHA-256:", "The complete registered report remains available", "Remove final receipt hashes", "support")
support = drop_paragraph(support, r"The package \path{output/article1_adaptive_revision", "Remove local network package path and editorial gate", "support")

# Visible numbers change mechanically; wording, mathematical values, captions,
# author details, and retained table cells are not edited.
heading_map = {}
sub_counts = {}
for match in re.finditer(r"\\subsection\*\{(\d+)\.(\d+) ([^\n]*)\}", main):
    sec, old = match.group(1), match.group(2)
    sub_counts[sec] = sub_counts.get(sec, 0) + 1
    heading_map[f"{sec}.{old}"] = f"{sec}.{sub_counts[sec]}"


def renumber_sections(text):
    text = re.sub(r"(\\subsection\*\{|\\addcontentsline\{toc\}\{subsection\}\{)(\d+\.\d+) ",
                  lambda m: m[1] + heading_map.get(m[2], m[2]) + " ", text)
    text = re.sub(r"(Section )(\d+\.\d+)", lambda m: m[1] + heading_map.get(m[2], m[2]), text)
    return text


def caption_ids(text, kind):
    prefix = "tab" if kind == "table" else "fig"
    return [int(x) for x in re.findall(r"\\caption\{[^\n]*?\\label\{" + prefix + r":(\d+)\}", text)]


main_tables = {old: str(i) for i, old in enumerate(caption_ids(main, "table"), 1)}
main_figures = {old: str(i) for i, old in enumerate(caption_ids(main, "figure"), 1)}
support_tables = {old: f"S{i}" for i, old in enumerate(caption_ids(support, "table"), 1)}
support_figures = {old: f"S{i}" for i, old in enumerate(caption_ids(support, "figure"), 1)}


def renumber_objects(text, tables, figures):
    text = re.sub(r"\bTable (\d+)\b", lambda m: "Table " + tables.get(int(m[1]), m[1]), text)
    text = re.sub(r"\bFig\. (\d+)\b", lambda m: "Fig. " + figures.get(int(m[1]), m[1]), text)
    return text


main = renumber_objects(renumber_sections(main), main_tables, main_figures)
support = renumber_objects(renumber_sections(support), main_tables | support_tables,
                           main_figures | support_figures)
support = support.replace(r"\begin{document}", r"\renewcommand{\thetable}{S\arabic{table}}" + "\n" +
                          r"\renewcommand{\thefigure}{S\arabic{figure}}" + "\n" + r"\begin{document}")


def prune_bibliography(text, document):
    begin = text.index(BIB)
    end = text.index(r"\end{thebibliography}")
    used = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", text[:begin]):
        used.update(group.split(","))
    bibliography = text[begin:end]
    for item in re.finditer(r"\\bibitem\{([^}]+)\}[^\n]*\n(?:\n)?", bibliography):
        if item[1] not in used:
            text = delete(text, item[0], "Remove uncited bibliography entry " + item[1], document)
    return text


main = prune_bibliography(main, "main")
support = prune_bibliography(support, "support")
(ROOT / "main.tex").write_text(main)
(ROOT / "supporting_appendices.tex").write_text(support)
manifest = {
    "source": "original_full.tex", "source_sha256": sha(SOURCE),
    "operation": "Deletion-only shortening; retained prose is not paraphrased",
    "main_deletion_only_before_numbering": is_subsequence(main_before_numbering, SOURCE),
    "main_pre_numbering_sha256": sha(main_before_numbering),
    "main_sha256": sha(main), "support_sha256": sha(support),
    "main_section_number_map": heading_map, "main_table_number_map": main_tables,
    "main_figure_number_map": main_figures, "support_table_number_map": support_tables,
    "support_figure_number_map": support_figures,
    "technical_exceptions": ["Mechanical heading and cross-reference renumbering",
                             "Supporting-PDF subtitle and S-prefixed object counters"],
    "preserved": ["Original full source unchanged", "All retained numerical results and table cells",
                  "144-source continuation results", "16-source 963-target negative adaptive comparison",
                  "Main network comparison", "Technical amendment disclosure in supporting Appendix L",
                  "Author list, abstract, funding, and availability statement"],
    "removed_characters_main_before_numbering": len(SOURCE) - len(main_before_numbering),
    "changes": changes,
}
(ROOT / "edit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"main_characters": len(main), "support_characters": len(support),
                  "main_tables": main_tables, "main_figures": main_figures,
                  "deleted_main_characters": manifest["removed_characters_main_before_numbering"],
                  "deletion_only": manifest["main_deletion_only_before_numbering"]}, indent=2))
