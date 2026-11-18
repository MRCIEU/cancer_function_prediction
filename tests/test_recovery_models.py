"""Template propagation matches the uniform walk when every edge weight is equal."""

from __future__ import annotations

import unittest

import torch

from src.recovery_models import propagate_templates


class TemplatePropagationTests(unittest.TestCase):
    def test_equal_weights_match_the_uniform_walk(self):
        # C1, C2, C3, G and cancers Q, K, M. Destinations of gene-gene steps are candidates.
        gc_gene = torch.tensor([3, 0, 0, 1, 3, 2, 3])
        gc_cancer = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        gg_src = torch.tensor([3, 3, 0, 2])
        gg_dst = torch.tensor([0, 1, 2, 0])
        two, three = propagate_templates(
            torch.ones(7),
            torch.ones(4),
            gc_gene,
            gc_cancer,
            gg_src,
            gg_dst,
            torch.tensor([0]),
            torch.tensor([True, True, True, False]),
            4,
            3,
        )
        self.assertEqual(two[0, :3].tolist(), [0.25, 0.25, 0.5])
        self.assertEqual(three[0, :3].tolist(), [0.125, 0.625, 0.25])


if __name__ == "__main__":
    unittest.main()
