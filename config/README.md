# Pinned sources and the frozen study rules

Records the source versions, the entity boundary, the path templates, and the development score.

Method: `data_sources.yaml` names the releases, checksums, and published row counts. `study_design.yaml` is the rule set the code is written against. The UMLS key is read from the environment at download time and is not stored in either file.

| File | Fixes |
| --- | --- |
| data_sources.yaml | CORE 1.1, SemMedDB semmedVER43_2024, NCBI Gene, iCite 2024-08 |
| study_design.yaml | Entity boundary, predicate allow-lists, two path templates, macro Recall@50 |

Previous result directories are not inputs. Downloading the same public release again does not erase earlier exposure.
