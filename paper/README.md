# arXiv Paper Package

**Title:** *Evidence-Bound Control for Verifiable Tool-Using Agents: A Stateful E-Commerce Customer-Service Study*  
**Author:** Rui Zou  
**Affiliation:** Harbin Institute of Technology  
**Email:** `25S012045@stu.hit.edu.cn`  
**Primary category:** `cs.AI`  
**Suggested cross-list:** `cs.CL`

## Files

- `evidence_bound_control.tex` — self-contained manuscript with numeric references and TikZ figures.
- `evidence_bound_control.pdf` — compiled 21-page preview.
- `references.bib` — verified BibTeX database for 19 primary sources.
- `PAPER_CONFIGURATION.md` — approved scope and evidence boundaries.
- `outline_and_argument.md` — approved outline and claim–evidence map.
- `evidence_audit.md` — repository evidence audit.
- `literature_report.md` — primary-source literature review.
- `WRITING_PRECOMMITMENT.md` — writing-quality contract used before drafting.

## Build

```powershell
pdflatex -interaction=nonstopmode -halt-on-error evidence_bound_control.tex
pdflatex -interaction=nonstopmode -halt-on-error evidence_bound_control.tex
```

The manuscript contains a self-contained `thebibliography` block so it can compile as a single file. `references.bib` is provided for migration to a conference template or a BibTeX-based arXiv source workflow.

## Submission notes

- The headline 589/600 result is a **development-suite system result** for frozen SFT plus controller r6.
- The corrected frozen-SFT model result is 530/600 with 13 hard-safety failures.
- The 508-to-530 change is evaluator rescoring of the same stored rollouts.
- The v18 GRPO calibration is 80/80 with 0% mixed groups; the formal RL gate is false and no policy update occurred.
- Before submission, add an ORCID if desired and ensure that external result/model archives referenced by hashes are accessible under an appropriate release policy.
