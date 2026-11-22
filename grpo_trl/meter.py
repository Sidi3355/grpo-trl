"""Compute accounting, identical for every optimiser.

A forward row of a network costs its multiply-accumulates (MACs: the sum of
``in_features * out_features`` over its linear layers). A row that is also
back-propagated costs three times that (forward plus roughly two forwards'
worth of backward). Rows are counted by forward hooks, so whatever code calls
a network, library code included, is charged the same way.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

TRAIN_ROW_FACTOR = 3.0


def macs(network: nn.Module) -> int:
    """Multiply-accumulates of one forward row."""
    return sum(
        layer.in_features * layer.out_features
        for layer in network.modules()
        if isinstance(layer, nn.Linear)
    )


@dataclass
class Meter:
    """Counts rows through the networks attached to it."""

    forward_rows: dict[str, int] = field(default_factory=dict)
    train_rows: dict[str, int] = field(default_factory=dict)
    macs: dict[str, int] = field(default_factory=dict)
    transitions: int = 0
    episodes: int = 0
    enabled: bool = True
    _handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def attach(self, name: str, network: nn.Module) -> None:
        """Charge every forward pass of ``network`` to ``name``."""
        self.forward_rows.setdefault(name, 0)
        self.train_rows.setdefault(name, 0)
        self.macs[name] = macs(network)

        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _out: object) -> None:
            if not self.enabled:
                return
            rows = int(inputs[0].shape[0])
            if torch.is_grad_enabled():
                self.train_rows[name] += rows
            else:
                self.forward_rows[name] += rows

        self._handles.append(network.register_forward_hook(hook))

    def detach(self) -> None:
        """Remove the hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def compute(self) -> float:
        """Total MACs spent so far."""
        return float(
            sum(
                self.macs[name]
                * (self.forward_rows[name] + TRAIN_ROW_FACTOR * self.train_rows[name])
                for name in self.macs
            )
        )

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Evaluation is not training: nothing inside is charged."""
        previous, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = previous

    def snapshot(self) -> dict[str, float]:
        """The counters, for a checkpoint record."""
        return {
            "compute": self.compute,
            "transitions": self.transitions,
            "episodes": self.episodes,
            **{f"fwd_{k}": v for k, v in self.forward_rows.items()},
            **{f"train_{k}": v for k, v in self.train_rows.items()},
        }
