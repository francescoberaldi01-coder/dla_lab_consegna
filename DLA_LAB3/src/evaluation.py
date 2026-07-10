"""Funzioni di valutazione per agenti di Reinforcement Learning."""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import torch

from .reinforce import select_action


def evaluate_policy(
    env: gym.Env,
    policy: torch.nn.Module,
    device: torch.device,
    num_episodes: int = 10,
    max_steps: int = 500,
    deterministic: bool = True,
) -> dict[str, float | list[float] | list[int]]:
    """Valuta una policy su piu' episodi senza aggiornare i pesi, tracciando reward e passi.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1); in
        01_reinforce_value_baseline (Esercizio 2); in 01_a2c_cartpole e
        02_a2c_lunarlander (Esercizio 3.1).
    """
    policy.eval()
    rewards: list[float] = []
    lengths: list[int] = []

    with torch.no_grad():
        for _ in range(num_episodes):
            observation, _ = env.reset()
            total_reward = 0.0
            episode_length = 0

            for _ in range(max_steps):
                action, _ = select_action(
                    policy,
                    observation,
                    device=device,
                    deterministic=deterministic,
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                episode_length += 1

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            lengths.append(episode_length)

    return {
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "lengths": lengths,
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": float(np.min(lengths)),
        "max_length": float(np.max(lengths)),
    }


def render_policy(
    environment_name: str,
    policy: torch.nn.Module,
    device: torch.device,
    num_episodes: int = 3,
    max_steps: int = 500,
    deterministic: bool = True,
    frame_delay: float = 0.02,
) -> list[float]:
    """Mostra la policy in tempo reale usando il renderer grafico di Gymnasium.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1); in
        01_reinforce_value_baseline (Esercizio 2); in 01_a2c_cartpole e
        02_a2c_lunarlander (Esercizio 3.1).
    """
    env = gym.make(environment_name, render_mode="human")
    policy.eval()
    rewards: list[float] = []

    try:
        with torch.no_grad():
            for _ in range(num_episodes):
                observation, _ = env.reset()
                total_reward = 0.0

                for _ in range(max_steps):
                    action, _ = select_action(
                        policy,
                        observation,
                        device=device,
                        deterministic=deterministic,
                    )
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)

                    # Il ritardo rende la finestra leggibile senza alterare la policy valutata.
                    if frame_delay > 0:
                        time.sleep(frame_delay)

                    if terminated or truncated:
                        break

                rewards.append(total_reward)
    finally:
        env.close()

    return rewards
