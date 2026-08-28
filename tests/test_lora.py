import unittest
from pathlib import Path
import sys

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.lora import (
    flatten_gradients,
    inject_lora,
    trainable_parameters,
)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 6, bias=False)
        self.v_proj = nn.Linear(4, 3, bias=False)
        self.out = nn.Linear(9, 2, bias=False)

    def forward(self, values):
        return self.out(torch.cat([self.q_proj(values), self.v_proj(values)], dim=-1))


class LoRATests(unittest.TestCase):
    def test_injection_preserves_initial_output_and_freezes_base(self):
        torch.manual_seed(7)
        model = TinyAttention()
        values = torch.randn(3, 4)
        expected = model(values).detach()
        targets = inject_lora(model, ["q_proj", "v_proj"], 2, 2, 11)
        self.assertEqual(targets, ["q_proj", "v_proj"])
        torch.testing.assert_close(model(values), expected)
        parameters = trainable_parameters(model)
        self.assertEqual(
            [name for name, _parameter in parameters],
            ["q_proj.lora_A", "q_proj.lora_B", "v_proj.lora_A", "v_proj.lora_B"],
        )
        self.assertTrue(all(not p.requires_grad for p in model.out.parameters()))

    def test_gradient_flattening_is_sorted_and_complete(self):
        model = TinyAttention()
        inject_lora(model, ["q_proj", "v_proj"], 2, 2, 13)
        model(torch.randn(2, 4)).sum().backward()
        parameters = trainable_parameters(model)
        gradient = flatten_gradients(parameters)
        self.assertEqual(gradient.dtype, torch.float32)
        self.assertEqual(gradient.numel(), sum(p.numel() for _, p in parameters))
        self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()

