"""Auxiliary functions for torch specifically."""
from typing import Any, Callable
import torch
import torch.nn as nn


def weight_initialisation(module: nn.Module, act_fn: Any) -> None:
    if isinstance(module, nn.Linear):
        match act_fn:
            case nn.ReLU() | nn.SELU() | nn.GELU() | nn.Mish():
                nn.init.kaiming_normal_(module.weight)
            case nn.Sigmoid() | nn.Tanh():
                nn.init.xavier_uniform_(module.weight)


def freeze_parameters(model: Any, excepted_layers: list[str]) -> None:
    for name, param in model.named_parameters():
        if sum([word in "".join(excepted_layers) for word in name.split(".")]) > 0:
            param.requires_grad = False


def get_trainable_parameters(model: Any) -> list[torch.Tensor]:
    return [param for _, param in model.named_parameters() if param.requires_grad]


def scale_to_range(x: Any, range: list[int] = [-1, 1]) -> Any:
    x_n = (x - x.min()) * (max(range) - min(range)) / (x.max() - x.min()) + min(range)
    return x_n


def get_laplacian(dim: int = 1) -> Callable:
    match dim:
        case 1:
            return laplacian1D
        case 2:
            return laplacian2D
        case 3:
            return laplacian3D
        case _:
            raise NotImplementedError("Dimension not implemented.")


def laplacian3D(mesh: torch.Tensor) -> torch.Tensor:
    alpha, beta = -0.3, 1.5
    xx, yy, zz = mesh[:, :, :, 0], mesh[:, :, :, 1], mesh[:, :, :, 2]
    x = torch.sqrt(xx**2 + yy**2 + zz**2)
    T = 1 - torch.exp(-torch.abs(x) ** alpha) ** beta
    T = scale_to_range(T, [0.2, 1.])
    return T

def laplacian2D(mesh: torch.Tensor) -> torch.Tensor:
    alpha, beta = -0.2, 1.5
    xx, yy = mesh[:, :, 0], mesh[:, :, 1]
    x = torch.sqrt(xx**2 + yy**2)
    T = 1 - torch.exp(-torch.abs(x) ** alpha) ** beta
    T = scale_to_range(T, [0.05, 1.])
    return T

def laplacian1D(mesh: torch.Tensor) -> torch.Tensor:
    alpha, beta = -0.3, 1.5
    xx = mesh[:, 0]
    x = torch.abs(xx)
    T = 1 - torch.exp(-torch.abs(x) ** alpha) ** beta
    T = scale_to_range(T, [0.2, 1.])
    return T
