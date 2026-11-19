# Joint-graph and recovery package

Builds the graph, audits the two path templates, prepares development inputs, and scores development recovery.

Method: Parsers keep identifiers, predicates, sentence labels, and PMIDs, and drop article and sentence prose. Path totals for the two templates are exact. Degree-weighted path counts use damping 0.4. Recovery reads a development allowlist supplied by the caller. Input preparation writes lists and features and does not fit a model.

| Entry | Does |
| --- | --- |
| prepare_data, align_entities, build_joint_graph | Pinned inputs, human-gene resolution, joint graph |
| extract_evidence_paths, path_evidence, complete_audit | Path counts, provenance, source sequences |
| citation_features, run_citation_audit | Citation motifs on adjacent edges |
| relation_attributes, relation_representation | Attribute splits and predicate direction |
| run_association_recovery, run_recovery_model | Development split and the six scorers |
| run_association_input_preparation | Frozen lists, masks, and feature columns |

A negated SemMedDB predicate stays negated and is not used as a path edge. Slurm entry points are the `submit_*.sbatch` files in the repository root. Each one calls the module named in that file.
