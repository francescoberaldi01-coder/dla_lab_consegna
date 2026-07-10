"""Utilita' generali per gli esperimenti di Deep Reinforcement Learning."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Imposta i seed principali per rendere gli esperimenti piu' riproducibili.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1); in
        01_reinforce_value_baseline (Esercizio 2); in 01_a2c_cartpole e
        02_a2c_lunarlander (Esercizio 3.1); in 01_dql_cartpole e
        02_dql_lunarlander (Esercizio 3.2).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_mps: bool = True, force_cpu: bool = False) -> torch.device:
    """Restituisce il miglior device PyTorch disponibile su Windows, macOS o CPU.

    Per ambienti piccoli come CartPole, impostare force_cpu=True evita il costo
    di coordinamento con GPU o MPS e puo' rendere il training piu' rapido.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1); in
        01_reinforce_value_baseline (Esercizio 2); in 01_a2c_cartpole,
        02_a2c_lunarlander e 03_a2c_lunarlander_hyperparameter_search
        (Esercizio 3.1); in 01_dql_cartpole e 02_dql_lunarlander (Esercizio 3.2).
    """
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def seed_environment(env, seed: int = 42):
    """Resetta un ambiente Gymnasium con seed e prova a fissare anche l'action space.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1); in
        01_reinforce_value_baseline (Esercizio 2); in 01_a2c_cartpole e
        02_a2c_lunarlander (Esercizio 3.1); in 01_dql_cartpole e
        02_dql_lunarlander (Esercizio 3.2).
    """
    observation, info = env.reset(seed=seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    return observation, info


def make_seeded_env(environment_name: str, seed: int):
    """Crea una factory per ambienti indipendenti usati nei vector env."""
    def _make_env():
        env = gym.make(environment_name)
        seed_environment(env, seed)
        return env

    return _make_env


def make_vector_env(
    environment_name: str,
    seed: int,
    num_envs: int,
    vector_env_type: str = "sync",
) -> gym.vector.VectorEnv:
    """Crea un vector env sincrono o asincrono con seed diversi per ogni copia.

    Note:
        Richiamata in 01_a2c_cartpole e 02_a2c_lunarlander (Esercizio 3.1); in
        01_dql_cartpole e 02_dql_lunarlander (Esercizio 3.2).
    """
    vector_env_class = (
        gym.vector.AsyncVectorEnv
        if vector_env_type == "async"
        else gym.vector.SyncVectorEnv
    )
    return vector_env_class(
        [
            make_seeded_env(environment_name, seed + env_idx)
            for env_idx in range(num_envs)
        ]
    )


def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copia lo stato di un modello su CPU per conservarlo come checkpoint leggero."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }


def restore_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor] | None) -> None:
    """Ripristina uno stato salvato, se disponibile, senza modificare altro.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1); in
        01_reinforce_value_baseline (Esercizio 2); in 01_a2c_cartpole e
        02_a2c_lunarlander (Esercizio 3.1); in 01_dql_cartpole e
        02_dql_lunarlander (Esercizio 3.2).
    """
    if state is not None:
        model.load_state_dict(state)


def save_checkpoint(
    state: dict[str, torch.Tensor] | torch.nn.Module,
    path: str,
) -> None:
    """Salva su disco uno stato di modello o un'istanza PyTorch (.pt/.pth)."""
    from pathlib import Path

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(state, torch.nn.Module):
        payload = state.state_dict()
    else:
        payload = state

    torch.save(payload, target_path)


def load_checkpoint(
    path: str,
    map_location: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Carica da disco un checkpoint precedentemente salvato."""
    from pathlib import Path

    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {target_path}")

    return torch.load(target_path, map_location=map_location)


def save_experiment_results(results: dict, path: str) -> None:
    """Esporta un dizionario di risultati o metriche in formato JSON riproducibile."""
    import json
    from pathlib import Path

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    def _default_encoder(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Oggetto di tipo {type(obj)} non serializzabile in JSON.")

    with open(target_path, "w", encoding="utf-8") as file_out:
        json.dump(results, file_out, indent=2, default=_default_encoder)


def load_experiment_results(path: str) -> dict:
    """Carica un dizionario di risultati precedentemente salvato in JSON."""
    import json
    from pathlib import Path

    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"File di risultati non trovato: {target_path}")

    with open(target_path, "r", encoding="utf-8") as file_in:
        return json.load(file_in)
