# Joint graph and evidence audit

This repository builds a provenance-preserving joint graph from the CORE Gene Expression–Cancer Knowledge Base and SemMedDB, then audits the two short path templates below. It does not train a model, rank candidates, or adjudicate article prose.

## Glossary

- `gene_cancer_association`: a CORE record linking one gene node to one cancer node. Sentence-level labels stay on the source record and are not stored as a single property of the pair.
- `semantic_predication`: a SemMedDB predication. The predicate, asserted subject and object, and negation flag are kept. A `NEG_` predicate is not rewritten as the opposite predicate.
- `source_article`: one PMID. Bibliographic fields come from CORE article metadata or the SemMedDB CITATIONS table.
- `citation_link`: a directed pair `citing_pmid` to `cited_pmid`. It is not a citation location and it is not evidence about why the citation was made.
- `connectivity_only`: an unordered pair of canonical entity identifiers plus the family gene-gene or gene-cancer. Predicate, direction, negation, and CORE attributes are omitted.
- `representation_audit`: a comparison, on the paths allowed by the study design, of that connectivity-only edge with the relation attributes and predication details that remain available.
- `one_intermediate_path`: candidate gene — intermediate gene — query cancer. This template contains two edges.
- `two_intermediate_path`: candidate gene — intermediate cancer — intermediate gene — query cancer. This template contains three edges.
- PMID: PubMed identifier. CUI: UMLS concept unique identifier.

## Layout

Code and configuration live in this repository. Pinned inputs and formal outputs live under `/home/b5ac/freddieyu.b5ac/projects/project_2`. Scratch holds downloads in progress and columnar intermediates.

The frozen rules are `config/study_design.yaml`. Source versions are `config/data_sources.yaml`.

## Run

Use the installed `hf-hpc` environment. Do not install packages for this phase.

```bash
conda activate hf-hpc
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_2
python -m unittest tests.test_graph_audit
sbatch submit_graph_audit.sbatch
```

`MODE=dry` measures a capped parse after the download. `MODE=formal` publishes `results/joint_graph_and_evidence_audit_<UTC>/`. The SemMedDB download uses `UMLS_API_KEY` from the environment and does not write that key to the repository, manifests, or logs.
