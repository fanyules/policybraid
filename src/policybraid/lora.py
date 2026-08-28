from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        a = torch.empty((rank, base.in_features), dtype=torch.float32, device="cpu")
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        b = torch.zeros((base.out_features, rank), dtype=torch.float32, device="cpu")
        self.lora_A = nn.Parameter(a.to(base.weight.device))
        self.lora_B = nn.Parameter(b.to(base.weight.device))
        self.scale = float(alpha) / rank

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        hidden = F.linear(inputs.to(self.lora_A.dtype), self.lora_A)
        delta = F.linear(hidden, self.lora_B) * self.scale
        return base_output + delta.to(base_output.dtype)


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    seed: int,
) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    replacements: list[tuple[nn.Module, str, nn.Linear, str]] = []
    targets = set(target_modules)
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child_name in targets:
                if not isinstance(child, nn.Linear):
                    raise TypeError(
                        f"target {module_name}.{child_name} is not torch.nn.Linear"
                    )
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                replacements.append((module, child_name, child, full_name))
    if not replacements:
        raise ValueError("no LoRA target modules were found")
    for parent, child_name, child, _full_name in replacements:
        setattr(parent, child_name, LoRALinear(child, rank, alpha, generator))
    return sorted(full_name for *_rest, full_name in replacements)


def trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    parameters = sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def flatten_gradients(
    parameters: list[tuple[str, nn.Parameter]],
) -> torch.Tensor:
    flattened = []
    for _name, parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter, dtype=torch.float32)
        flattened.append(gradient.detach().to(device="cpu", dtype=torch.float32).reshape(-1))
    return torch.cat(flattened)


def parameter_manifest(
    parameters: list[tuple[str, nn.Parameter]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "elements": parameter.numel(),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in parameters
    ]

