"""Implementazione modulare dell'algoritmo REINFORCE."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Optimizer


@dataclass
class Episode:
    """Dati raccolti durante un episodio completo."""

    observations: list[np.ndarray]
    rewards: list[float]
    log_probs: list[torch.Tensor]
    total_reward: float


@dataclass
class TrainingResult:
    """Metriche essenziali prodotte dal training di REINFORCE."""

    episode_rewards: list[float]
    episode_losses: list[float]
    eval_episodes: list[int] | None = None
    eval_mean_rewards: list[float] | None = None
    eval_mean_lengths: list[float] | None = None
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_policy_state: dict[str, torch.Tensor] | None = None


@dataclass
class BaselineTrainingResult:
    """Metriche prodotte dal training di REINFORCE con baseline di valore."""

    episode_rewards: list[float]
    policy_losses: list[float]
    value_losses: list[float]
    eval_episodes: list[int] | None = None
    eval_mean_rewards: list[float] | None = None
    eval_mean_lengths: list[float] | None = None
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_policy_state: dict[str, torch.Tensor] | None = None
    best_value_state: dict[str, torch.Tensor] | None = None


def select_action(
    policy: nn.Module,
    observation: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
) -> tuple[int, torch.Tensor]:
    """Seleziona un'azione dalla policy e restituisce anche la sua log-probabilita'."""
    state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    logits = policy(state)
    distribution = Categorical(logits=logits)

    if deterministic:
        action = torch.argmax(logits, dim=-1)
    else:
        action = distribution.sample()

    log_prob = distribution.log_prob(action).squeeze(0)
    return int(action.item()), log_prob


def compute_discounted_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    """Calcola il ritorno scontato G_t per ogni passo di un episodio.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1).
    """
    discounted_returns: list[float] = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        discounted_returns.append(running_return)

    discounted_returns.reverse()
    return torch.tensor(discounted_returns, dtype=torch.float32)


def run_episode(
    env: gym.Env,
    policy: nn.Module,
    device: torch.device,
    max_steps: int = 500,
) -> Episode:
    """Esegue un episodio usando la policy corrente e salva reward e log-probabilita'."""
    observation, _ = env.reset()
    observations: list[np.ndarray] = []
    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []

    for _ in range(max_steps):
        observations.append(np.asarray(observation, dtype=np.float32))
        action, log_prob = select_action(policy, observation, device=device)
        observation, reward, terminated, truncated, _ = env.step(action)

        rewards.append(float(reward))
        log_probs.append(log_prob)

        if terminated or truncated:
            break

    return Episode(
        observations=observations,
        rewards=rewards,
        log_probs=log_probs,
        total_reward=float(sum(rewards)),
    )


def reinforce_loss(
    episode: Episode,
    gamma: float = 0.99,
    standardize_returns: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Costruisce la loss policy-gradient per un episodio."""
    returns = compute_discounted_returns(episode.rewards, gamma=gamma)

    if standardize_returns and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

    if device is not None:
        returns = returns.to(device)

    log_probs = torch.stack(episode.log_probs)
    return -(log_probs * returns).sum()


def train_reinforce(
    env: gym.Env,
    policy: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_episodes: int = 300,
    gamma: float = 0.99,
    max_steps: int = 500,
    standardize_returns: bool = True,
    eval_env: gym.Env | None = None,
    eval_every_n: int | None = None,
    eval_episodes_m: int = 10,
) -> TrainingResult:
    """Addestra una policy con REINFORCE e restituisce reward e loss per episodio.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1); in
        01_reinforce_value_baseline (Esercizio 2).
    """
    policy.train()
    episode_rewards: list[float] = []
    episode_losses: list[float] = []

    eval_episodes: list[int] = []
    eval_mean_rewards: list[float] = []
    eval_mean_lengths: list[float] = []
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_policy_state: dict[str, torch.Tensor] | None = None

    for episode_idx in range(1, num_episodes + 1):
        policy.train()
        episode = run_episode(env, policy, device=device, max_steps=max_steps)
        loss = reinforce_loss(
            episode,
            gamma=gamma,
            standardize_returns=standardize_returns,
            device=device,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(episode.total_reward)
        episode_losses.append(float(loss.detach().cpu().item()))

        # Valutazione periodica
        if eval_env is not None and eval_every_n is not None and episode_idx % eval_every_n == 0:
            from .evaluation import evaluate_policy

            eval_res = evaluate_policy(
                env=eval_env,
                policy=policy,
                device=device,
                num_episodes=eval_episodes_m,
                max_steps=max_steps,
                deterministic=True,
            )
            eval_episodes.append(episode_idx)
            eval_mean_rewards.append(eval_res["mean_reward"])
            eval_mean_lengths.append(eval_res["mean_length"])
            if best_eval_mean_reward is None or eval_res["mean_reward"] > best_eval_mean_reward:
                from .utils import clone_model_state

                best_eval_episode = episode_idx
                best_eval_mean_reward = eval_res["mean_reward"]
                best_policy_state = clone_model_state(policy)

    return TrainingResult(
        episode_rewards=episode_rewards,
        episode_losses=episode_losses,
        eval_episodes=eval_episodes if eval_every_n is not None else None,
        eval_mean_rewards=eval_mean_rewards if eval_every_n is not None else None,
        eval_mean_lengths=eval_mean_lengths if eval_every_n is not None else None,
        best_eval_episode=best_eval_episode,
        best_eval_mean_reward=best_eval_mean_reward,
        best_policy_state=best_policy_state,
    )


def reinforce_with_value_baseline_loss(
    episode: Episode,
    value_network: nn.Module,
    gamma: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcola loss della policy e loss della rete di valore per un episodio."""
    returns = compute_discounted_returns(episode.rewards, gamma=gamma).to(device)
    observations = torch.as_tensor(np.asarray(episode.observations), dtype=torch.float32, device=device)
    values = value_network(observations)

    # L'advantage guida la policy, ma non deve aggiornare direttamente la rete di valore.
    advantages = returns - values.detach()
    log_probs = torch.stack(episode.log_probs)
    policy_loss = -(log_probs * advantages).sum()
    value_loss = nn.functional.mse_loss(values, returns)

    return policy_loss, value_loss


def train_reinforce_with_value_baseline(
    env: gym.Env,
    policy: nn.Module,
    value_network: nn.Module,
    policy_optimizer: Optimizer,
    value_optimizer: Optimizer,
    device: torch.device,
    num_episodes: int = 300,
    gamma: float = 0.99,
    max_steps: int = 500,
    value_loss_weight: float = 1.0,
    eval_env: gym.Env | None = None,
    eval_every_n: int | None = None,
    eval_episodes_m: int = 10,
) -> BaselineTrainingResult:
    """Addestra REINFORCE usando V(s) come baseline appresa.

    Note:
        Richiamata in 01_reinforce_value_baseline (Esercizio 2).
    """
    policy.train()
    value_network.train()
    episode_rewards: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []

    eval_episodes: list[int] = []
    eval_mean_rewards: list[float] = []
    eval_mean_lengths: list[float] = []
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_policy_state: dict[str, torch.Tensor] | None = None
    best_value_state: dict[str, torch.Tensor] | None = None

    for episode_idx in range(1, num_episodes + 1):
        policy.train()
        value_network.train()
        episode = run_episode(env, policy, device=device, max_steps=max_steps)
        policy_loss, value_loss = reinforce_with_value_baseline_loss(
            episode=episode,
            value_network=value_network,
            gamma=gamma,
            device=device,
        )

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        (value_loss_weight * value_loss).backward()
        value_optimizer.step()

        episode_rewards.append(episode.total_reward)
        policy_losses.append(float(policy_loss.detach().cpu().item()))
        value_losses.append(float(value_loss.detach().cpu().item()))

        # Valutazione periodica
        if eval_env is not None and eval_every_n is not None and episode_idx % eval_every_n == 0:
            from .evaluation import evaluate_policy

            eval_res = evaluate_policy(
                env=eval_env,
                policy=policy,
                device=device,
                num_episodes=eval_episodes_m,
                max_steps=max_steps,
                deterministic=True,
            )
            eval_episodes.append(episode_idx)
            eval_mean_rewards.append(eval_res["mean_reward"])
            eval_mean_lengths.append(eval_res["mean_length"])
            if best_eval_mean_reward is None or eval_res["mean_reward"] > best_eval_mean_reward:
                from .utils import clone_model_state

                best_eval_episode = episode_idx
                best_eval_mean_reward = eval_res["mean_reward"]
                best_policy_state = clone_model_state(policy)
                best_value_state = clone_model_state(value_network)

    return BaselineTrainingResult(
        episode_rewards=episode_rewards,
        policy_losses=policy_losses,
        value_losses=value_losses,
        eval_episodes=eval_episodes if eval_every_n is not None else None,
        eval_mean_rewards=eval_mean_rewards if eval_every_n is not None else None,
        eval_mean_lengths=eval_mean_lengths if eval_every_n is not None else None,
        best_eval_episode=best_eval_episode,
        best_eval_mean_reward=best_eval_mean_reward,
        best_policy_state=best_policy_state,
        best_value_state=best_value_state,
    )
