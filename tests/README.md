# CPU checks

Tests parsing, path weights, citation motifs, attribute splits, and development-recovery rules.

Method: Unittest against fixtures and hand-enumerated graphs. Synthetic rows stay inside the test. A passing test is not a recovery score.

| File | Covers |
| --- | --- |
| test_graph_audit.py | Parsing, mapping, path counts, provenance |
| test_path_evidence.py | Weights, overlap, provenance fixtures |
| test_complete_audit.py | Source sequences and candidate path counts |
| test_citation_features.py | Adjacent-edge motifs and traversal alignment |
| test_relation_attributes.py | DWPC weights and attribute splits |
| test_relation_representation.py | Same-sentence triples and traversal direction |
| test_association_recovery.py | Development-recovery rules |
| test_recovery_scope.py | Closed-form scores on enumerated paths |
| test_recovery_models.py | Template propagation against a uniform walk |
| test_input_preparation.py | Evaluation lists, citation states, path budget |

Formal test labels are not read. Gold for a recovery check is the allowlist passed into the function.
