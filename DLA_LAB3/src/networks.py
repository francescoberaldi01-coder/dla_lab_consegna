"""Reti neurali usate negli esercizi di Reinforcement Learning."""

from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """Policy network MLP per ambienti con stato continuo e azioni discrete.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1); in
        01_reinforce_value_baseline (Esercizio 2); in 01_a2c_cartpole e
        02_a2c_lunarlander (Esercizio 3.1).
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        """Inizializza una policy che produce i logit delle azioni disponibili."""
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Calcola i logit non normalizzati per ogni azione."""
        return self.model(observations)


class ValueNetwork(nn.Module):
    """Rete MLP che stima il valore atteso di uno stato.

    Note:
        Richiamata in 01_reinforce_value_baseline (Esercizio 2); in
        01_a2c_cartpole e 02_a2c_lunarlander (Esercizio 3.1).
    """

    def __init__(self, observation_dim: int, hidden_dim: int = 128) -> None:
        """Inizializza una rete che produce un singolo valore scalare per stato."""
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Restituisce la stima V(s) per ogni osservazione del batch."""
        return self.model(observations).squeeze(-1)


class QNetwork(nn.Module):
    """Rete MLP che stima Q(s, a) per tutte le azioni discrete.

    Note:
        Richiamata in 01_dql_cartpole e 02_dql_lunarlander (Esercizio 3.2).
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        """Inizializza una rete che produce un valore Q per ogni azione."""
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Restituisce i valori Q associati alle azioni disponibili."""
        return self.model(observations)
