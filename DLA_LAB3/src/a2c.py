"""Implementazione semplice di Advantage Actor-Critic per ambienti discreti."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Optimizer


@dataclass
class A2CRollout:
    """Dati raccolti in un rollout n-step usato per aggiornare actor e critic."""

    rewards: list[float]
    log_probs: list[torch.Tensor]
    entropies: list[torch.Tensor]
    values: list[torch.Tensor]
    final_observation: np.ndarray
    done: bool
    total_reward: float


@dataclass
class A2CVectorRollout:
    """Dati raccolti da piu' ambienti indipendenti nello stesso rollout."""

    rewards: list[torch.Tensor]
    dones: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    entropies: list[torch.Tensor]
    values: list[torch.Tensor]
    final_observations: np.ndarray


@dataclass
class A2CTrainingResult:
    """Metriche essenziali prodotte dal training A2C."""

    episode_rewards: list[float]
    actor_losses: list[float]
    critic_losses: list[float]
    total_losses: list[float]
    eval_episodes: list[int] | None = None
    eval_mean_rewards: list[float] | None = None
    eval_mean_lengths: list[float] | None = None
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_actor_state: dict[str, torch.Tensor] | None = None
    best_critic_state: dict[str, torch.Tensor] | None = None


EvalCallback = Callable[[int, dict[str, float | list[float] | list[int]]], None]


class ReturnNormalizer:
    """Stima progressiva (media mobile) della varianza dei ritorni bootstrap.

    Usata solo per riscalare la loss del critic: a differenza della standardizzazione
    dell'advantage, qui la stima deve accumularsi su molti rollout invece che resettarsi
    ad ogni batch, altrimenti si cancellerebbe l'informazione di scala che il critic deve
    imparare (rollout_steps e' troppo piccolo per una stima affidabile batch per batch).
    """

    def __init__(self, decay: float = 0.99, epsilon: float = 1e-8):
        self.decay = decay
        self.epsilon = epsilon
        self.running_var = 1.0

    def update(self, returns: torch.Tensor) -> None:
        """Aggiorna la stima esponenziale della varianza dei ritorni."""
        batch_var = float(returns.var(unbiased=False).item())
        self.running_var = self.decay * self.running_var + (1.0 - self.decay) * batch_var

    @property
    def std(self) -> float:
        """Restituisce la deviazione standard corrente con stabilizzazione numerica."""
        return self.running_var**0.5 + self.epsilon


def select_a2c_action(
    actor: nn.Module,
    critic: nn.Module,
    observation: np.ndarray,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Campiona un'azione e restituisce log-probabilita', entropia e valore."""
    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    logits = actor(state)
    distribution = Categorical(logits=logits)
    action = distribution.sample()
    value = critic(state).squeeze(0)

    return (
        int(action.item()),
        distribution.log_prob(action).squeeze(0),
        distribution.entropy().squeeze(0),
        value,
    )


def select_a2c_actions(
    actor: nn.Module,
    critic: nn.Module,
    observations: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Campiona azioni per un batch di osservazioni vettorializzate."""
    states = torch.as_tensor(
        observations,
        dtype=torch.float32,
        device=device,
    )
    logits = actor(states)
    distribution = Categorical(logits=logits)
    actions = distribution.sample()
    values = critic(states)

    return (
        actions.detach().cpu().numpy(),
        distribution.log_prob(actions),
        distribution.entropy(),
        values,
    )


def collect_rollout(
    env: gym.Env,
    actor: nn.Module,
    critic: nn.Module,
    observation: np.ndarray,
    device: torch.device,
    rollout_steps: int,
) -> A2CRollout:
    """Raccoglie al massimo `rollout_steps` transizioni dalla policy corrente."""
    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    done = False
    current_observation = observation

    for _ in range(rollout_steps):
        action, log_prob, entropy, value = select_a2c_action(
            actor=actor,
            critic=critic,
            observation=current_observation,
            device=device,
        )
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rewards.append(float(reward))
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        current_observation = next_observation

        if done:
            break

    return A2CRollout(
        rewards=rewards,
        log_probs=log_probs,
        entropies=entropies,
        values=values,
        final_observation=np.asarray(current_observation, dtype=np.float32),
        done=bool(done),
        total_reward=float(sum(rewards)),
    )


def collect_vector_rollout(
    envs: gym.vector.VectorEnv,
    actor: nn.Module,
    critic: nn.Module,
    observations: np.ndarray,
    device: torch.device,
    rollout_steps: int,
) -> A2CVectorRollout:
    """Raccoglie rollout sincroni da un insieme di ambienti vettorializzati."""
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    current_observations = observations

    for _ in range(rollout_steps):
        actions, log_prob, entropy, value = select_a2c_actions(
            actor=actor,
            critic=critic,
            observations=current_observations,
            device=device,
        )
        next_observations, reward, terminated, truncated, _ = envs.step(actions)
        done = np.logical_or(terminated, truncated)

        rewards.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
        dones.append(torch.as_tensor(done, dtype=torch.float32, device=device))
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        current_observations = next_observations

    return A2CVectorRollout(
        rewards=rewards,
        dones=dones,
        log_probs=log_probs,
        entropies=entropies,
        values=values,
        final_observations=np.asarray(current_observations, dtype=np.float32),
    )


def compute_bootstrapped_returns(
    rollout: A2CRollout,
    critic: nn.Module,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Calcola ritorni n-step usando V(s) come bootstrap quando serve."""
    if rollout.done:
        running_return = torch.tensor(0.0, dtype=torch.float32, device=device)
    else:
        final_state = torch.as_tensor(
            rollout.final_observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            running_return = critic(final_state).squeeze(0)

    returns: list[torch.Tensor] = []
    for reward in reversed(rollout.rewards):
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
        running_return = reward_tensor + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return torch.stack(returns)


def compute_vector_bootstrapped_returns(
    rollout: A2CVectorRollout,
    critic: nn.Module,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Calcola ritorni n-step per rollout raccolti da piu' ambienti."""
    final_states = torch.as_tensor(
        rollout.final_observations,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        running_returns = critic(final_states)

    returns: list[torch.Tensor] = []
    for reward, done in zip(reversed(rollout.rewards), reversed(rollout.dones)):
        running_returns = reward + gamma * running_returns * (1.0 - done)
        returns.append(running_returns)

    returns.reverse()
    return torch.stack(returns)


def a2c_loss(
    rollout: A2CRollout,
    critic: nn.Module,
    gamma: float,
    device: torch.device,
    value_loss_weight: float = 1.0,
    standardize_advantages: bool = False,
    entropy_weight: float = 0.0,
    critic_normalizer: ReturnNormalizer | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcola loss dell'actor, loss del critic e loss totale."""
    returns = compute_bootstrapped_returns(
        rollout=rollout,
        critic=critic,
        gamma=gamma,
        device=device,
    )
    values = torch.stack(rollout.values)
    log_probs = torch.stack(rollout.log_probs)
    entropies = torch.stack(rollout.entropies)
    advantages = returns - values

    if standardize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

    actor_loss = -(log_probs * advantages.detach()).mean()

    if critic_normalizer is not None:
        critic_normalizer.update(returns.detach())
        scale = critic_normalizer.std
        critic_loss = nn.functional.mse_loss(values / scale, returns.detach() / scale)
    else:
        critic_loss = nn.functional.mse_loss(values, returns.detach())

    entropy_bonus = entropies.mean()
    total_loss = (
        actor_loss
        + value_loss_weight * critic_loss
        - entropy_weight * entropy_bonus
    )

    return actor_loss, critic_loss, total_loss


def vectorized_a2c_loss(
    rollout: A2CVectorRollout,
    critic: nn.Module,
    gamma: float,
    device: torch.device,
    value_loss_weight: float = 1.0,
    standardize_advantages: bool = False,
    entropy_weight: float = 0.0,
    critic_normalizer: ReturnNormalizer | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcola la loss A2C aggregando rollout raccolti in parallelo."""
    returns = compute_vector_bootstrapped_returns(
        rollout=rollout,
        critic=critic,
        gamma=gamma,
        device=device,
    )
    values = torch.stack(rollout.values)
    log_probs = torch.stack(rollout.log_probs)
    entropies = torch.stack(rollout.entropies)
    advantages = returns - values

    flat_returns = returns.reshape(-1)
    flat_values = values.reshape(-1)
    flat_log_probs = log_probs.reshape(-1)
    flat_entropies = entropies.reshape(-1)
    flat_advantages = advantages.reshape(-1)

    if standardize_advantages and len(flat_advantages) > 1:
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std(unbiased=False) + 1e-8
        )

    actor_loss = -(flat_log_probs * flat_advantages.detach()).mean()

    if critic_normalizer is not None:
        critic_normalizer.update(flat_returns.detach())
        scale = critic_normalizer.std
        critic_loss = nn.functional.mse_loss(flat_values / scale, flat_returns.detach() / scale)
    else:
        critic_loss = nn.functional.mse_loss(flat_values, flat_returns.detach())

    entropy_bonus = flat_entropies.mean()
    total_loss = (
        actor_loss
        + value_loss_weight * critic_loss
        - entropy_weight * entropy_bonus
    )

    return actor_loss, critic_loss, total_loss


def train_a2c(
    env: gym.Env,
    actor: nn.Module,
    critic: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_episodes: int = 300,
    gamma: float = 0.99,
    max_steps: int = 500,
    rollout_steps: int = 5,
    value_loss_weight: float = 1.0,
    standardize_advantages: bool = False,
    standardize_critic_targets: bool = False,
    entropy_weight: float = 0.0,
    eval_env: gym.Env | None = None,
    eval_every_n: int | None = None,
    eval_episodes_m: int = 10,
    eval_callback: EvalCallback | None = None,
) -> A2CTrainingResult:
    """Addestra actor e critic con A2C e restituisce le metriche per episodio.

    Note:
        Richiamata in 01_a2c_cartpole e 02_a2c_lunarlander (Esercizio 3.1).
    """
    actor.train()
    critic.train()
    episode_rewards: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    total_losses: list[float] = []

    eval_episodes: list[int] = []
    eval_mean_rewards: list[float] = []
    eval_mean_lengths: list[float] = []
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_actor_state: dict[str, torch.Tensor] | None = None
    best_critic_state: dict[str, torch.Tensor] | None = None
    critic_normalizer = ReturnNormalizer() if standardize_critic_targets else None

    observation, _ = env.reset()
    current_episode_reward = 0.0
    completed_episodes = 0

    while completed_episodes < num_episodes:
        actor.train()
        critic.train()
        rollout = collect_rollout(
            env=env,
            actor=actor,
            critic=critic,
            observation=observation,
            device=device,
            rollout_steps=rollout_steps,
        )
        actor_loss, critic_loss, total_loss = a2c_loss(
            rollout=rollout,
            critic=critic,
            gamma=gamma,
            device=device,
            value_loss_weight=value_loss_weight,
            standardize_advantages=standardize_advantages,
            entropy_weight=entropy_weight,
            critic_normalizer=critic_normalizer,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        actor_losses.append(float(actor_loss.detach().cpu().item()))
        critic_losses.append(float(critic_loss.detach().cpu().item()))
        total_losses.append(float(total_loss.detach().cpu().item()))

        current_episode_reward += rollout.total_reward

        if rollout.done:
            completed_episodes += 1
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            observation, _ = env.reset()
        else:
            observation = rollout.final_observation
            continue

        # Valutazione periodica della sola policy actor, senza aggiornare i pesi.
        should_evaluate = (
            eval_env is not None
            and eval_every_n is not None
            and completed_episodes % eval_every_n == 0
        )
        if should_evaluate:
            from .evaluation import evaluate_policy

            eval_res = evaluate_policy(
                env=eval_env,
                policy=actor,
                device=device,
                num_episodes=eval_episodes_m,
                max_steps=max_steps,
                deterministic=True,
            )
            eval_episodes.append(completed_episodes)
            eval_mean_rewards.append(eval_res["mean_reward"])
            eval_mean_lengths.append(eval_res["mean_length"])
            if eval_callback is not None:
                eval_callback(completed_episodes, eval_res)
            if best_eval_mean_reward is None or eval_res["mean_reward"] > best_eval_mean_reward:
                from .utils import clone_model_state

                best_eval_episode = completed_episodes
                best_eval_mean_reward = eval_res["mean_reward"]
                best_actor_state = clone_model_state(actor)
                best_critic_state = clone_model_state(critic)

    return A2CTrainingResult(
        episode_rewards=episode_rewards,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        total_losses=total_losses,
        eval_episodes=eval_episodes if eval_every_n is not None else None,
        eval_mean_rewards=eval_mean_rewards if eval_every_n is not None else None,
        eval_mean_lengths=eval_mean_lengths if eval_every_n is not None else None,
        best_eval_episode=best_eval_episode,
        best_eval_mean_reward=best_eval_mean_reward,
        best_actor_state=best_actor_state,
        best_critic_state=best_critic_state,
    )


def train_vectorized_a2c(
    envs: gym.vector.VectorEnv,
    actor: nn.Module,
    critic: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_episodes: int = 300,
    gamma: float = 0.99,
    max_steps: int = 500,
    rollout_steps: int = 5,
    value_loss_weight: float = 1.0,
    standardize_advantages: bool = False,
    standardize_critic_targets: bool = False,
    entropy_weight: float = 0.0,
    eval_env: gym.Env | None = None,
    eval_every_n: int | None = None,
    eval_episodes_m: int = 10,
    eval_callback: EvalCallback | None = None,
) -> A2CTrainingResult:
    """Addestra A2C aggregando rollout sincroni da piu' ambienti indipendenti.

    Note:
        Richiamata in 01_a2c_cartpole e 02_a2c_lunarlander (Esercizio 3.1).
    """
    actor.train()
    critic.train()
    episode_rewards: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    total_losses: list[float] = []

    eval_episodes: list[int] = []
    eval_mean_rewards: list[float] = []
    eval_mean_lengths: list[float] = []
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_actor_state: dict[str, torch.Tensor] | None = None
    best_critic_state: dict[str, torch.Tensor] | None = None
    critic_normalizer = ReturnNormalizer() if standardize_critic_targets else None

    observations, _ = envs.reset()
    num_envs = int(envs.num_envs)
    current_episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed_episodes = 0

    while completed_episodes < num_episodes:
        actor.train()
        critic.train()
        rollout = collect_vector_rollout(
            envs=envs,
            actor=actor,
            critic=critic,
            observations=observations,
            device=device,
            rollout_steps=rollout_steps,
        )
        actor_loss, critic_loss, total_loss = vectorized_a2c_loss(
            rollout=rollout,
            critic=critic,
            gamma=gamma,
            device=device,
            value_loss_weight=value_loss_weight,
            standardize_advantages=standardize_advantages,
            entropy_weight=entropy_weight,
            critic_normalizer=critic_normalizer,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        actor_losses.append(float(actor_loss.detach().cpu().item()))
        critic_losses.append(float(critic_loss.detach().cpu().item()))
        total_losses.append(float(total_loss.detach().cpu().item()))

        for reward_tensor, done_tensor in zip(rollout.rewards, rollout.dones):
            reward_values = reward_tensor.detach().cpu().numpy()
            done_values = done_tensor.detach().cpu().numpy().astype(bool)
            current_episode_rewards += reward_values

            for env_idx, done in enumerate(done_values):
                if done and completed_episodes < num_episodes:
                    completed_episodes += 1
                    episode_rewards.append(float(current_episode_rewards[env_idx]))
                    current_episode_rewards[env_idx] = 0.0

                    should_evaluate = (
                        eval_env is not None
                        and eval_every_n is not None
                        and completed_episodes % eval_every_n == 0
                    )
                    if should_evaluate:
                        from .evaluation import evaluate_policy

                        eval_res = evaluate_policy(
                            env=eval_env,
                            policy=actor,
                            device=device,
                            num_episodes=eval_episodes_m,
                            max_steps=max_steps,
                            deterministic=True,
                        )
                        eval_episodes.append(completed_episodes)
                        eval_mean_rewards.append(eval_res["mean_reward"])
                        eval_mean_lengths.append(eval_res["mean_length"])
                        if eval_callback is not None:
                            eval_callback(completed_episodes, eval_res)
                        if best_eval_mean_reward is None or eval_res["mean_reward"] > best_eval_mean_reward:
                            from .utils import clone_model_state

                            best_eval_episode = completed_episodes
                            best_eval_mean_reward = eval_res["mean_reward"]
                            best_actor_state = clone_model_state(actor)
                            best_critic_state = clone_model_state(critic)

        observations = rollout.final_observations

    return A2CTrainingResult(
        episode_rewards=episode_rewards,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        total_losses=total_losses,
        eval_episodes=eval_episodes if eval_every_n is not None else None,
        eval_mean_rewards=eval_mean_rewards if eval_every_n is not None else None,
        eval_mean_lengths=eval_mean_lengths if eval_every_n is not None else None,
        best_eval_episode=best_eval_episode,
        best_eval_mean_reward=best_eval_mean_reward,
        best_actor_state=best_actor_state,
        best_critic_state=best_critic_state,
    )
