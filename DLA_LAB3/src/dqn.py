"""Implementazione essenziale di Deep Q-Learning per ambienti discreti."""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class Transition:
    """Singola transizione salvata nel replay buffer."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


@dataclass
class DQNTrainingResult:
    """Metriche essenziali prodotte dal training DQN."""

    episode_rewards: list[float]
    episode_lengths: list[int]
    losses: list[float]
    epsilons: list[float]
    eval_episodes: list[int] | None = None
    eval_mean_rewards: list[float] | None = None
    eval_mean_lengths: list[float] | None = None
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_q_state: dict[str, torch.Tensor] | None = None
    best_target_state: dict[str, torch.Tensor] | None = None


class ReplayBuffer:
    """Replay buffer FIFO con campionamento casuale uniforme."""

    def __init__(self, capacity: int) -> None:
        """Crea un buffer con capacita' massima fissata."""
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Restituisce il numero di transizioni attualmente disponibili."""
        return len(self.buffer)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Aggiunge una transizione al buffer."""
        self.buffer.append(
            Transition(
                observation=np.asarray(observation, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_observation=np.asarray(next_observation, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        """Estrae un minibatch casuale di transizioni."""
        return random.sample(self.buffer, batch_size)


def compute_epsilon(
    step: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> float:
    """Calcola epsilon con decadimento lineare fino al valore minimo."""
    if epsilon_decay_steps <= 0:
        return epsilon_end

    progress = min(step / epsilon_decay_steps, 1.0)
    return epsilon_start + progress * (epsilon_end - epsilon_start)


def select_dqn_action(
    q_network: nn.Module,
    observation: np.ndarray,
    device: torch.device,
    action_dim: int,
    epsilon: float = 0.0,
) -> int:
    """Seleziona un'azione con strategia epsilon-greedy."""
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = q_network(state)

    return int(torch.argmax(q_values, dim=1).item())


def select_dqn_actions(
    q_network: nn.Module,
    observations: np.ndarray,
    device: torch.device,
    action_dim: int,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Seleziona un batch di azioni con strategia epsilon-greedy."""
    actions: list[int] = []
    random_mask = np.random.random(size=len(observations)) < epsilon

    if np.any(~random_mask):
        states = torch.as_tensor(
            observations[~random_mask],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            greedy_actions = q_network(states).argmax(dim=1).detach().cpu().numpy()
    else:
        greedy_actions = np.array([], dtype=np.int64)

    greedy_idx = 0
    for use_random_action in random_mask:
        if use_random_action:
            actions.append(random.randrange(action_dim))
        else:
            actions.append(int(greedy_actions[greedy_idx]))
            greedy_idx += 1

    return np.asarray(actions, dtype=np.int64)


def compute_dqn_loss(
    q_network: nn.Module,
    target_network: nn.Module,
    transitions: list[Transition],
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Calcola la loss DQN usando la target network per il bootstrap."""
    observations = torch.as_tensor(
        np.asarray([transition.observation for transition in transitions]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        [transition.action for transition in transitions],
        dtype=torch.long,
        device=device,
    )
    rewards = torch.as_tensor(
        [transition.reward for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    next_observations = torch.as_tensor(
        np.asarray([transition.next_observation for transition in transitions]),
        dtype=torch.float32,
        device=device,
    )
    dones = torch.as_tensor(
        [transition.done for transition in transitions],
        dtype=torch.float32,
        device=device,
    )

    current_q_values = q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_observations).max(dim=1).values
        target_q_values = rewards + gamma * (1.0 - dones) * next_q_values

    return nn.functional.smooth_l1_loss(current_q_values, target_q_values)


def update_dqn_network(
    q_network: nn.Module,
    target_network: nn.Module,
    optimizer: Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    gradient_clip: float | None,
) -> float:
    """Esegue un singolo aggiornamento DQN e restituisce il valore della loss."""
    q_network.train()
    transitions = replay_buffer.sample(batch_size=batch_size)
    loss = compute_dqn_loss(
        q_network=q_network,
        target_network=target_network,
        transitions=transitions,
        gamma=gamma,
        device=device,
    )

    optimizer.zero_grad()
    loss.backward()
    if gradient_clip is not None:
        nn.utils.clip_grad_norm_(q_network.parameters(), gradient_clip)
    optimizer.step()

    return float(loss.detach().cpu().item())


def train_dqn(
    env: gym.Env,
    q_network: nn.Module,
    target_network: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_episodes: int = 300,
    gamma: float = 0.99,
    max_steps: int = 500,
    batch_size: int = 64,
    replay_capacity: int = 50_000,
    min_replay_size: int = 1_000,
    train_every_steps: int = 1,
    target_update_every_steps: int = 500,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 10_000,
    gradient_clip: float | None = 10.0,
    eval_env: gym.Env | None = None,
    eval_every_n: int | None = None,
    eval_episodes_m: int = 10,
) -> DQNTrainingResult:
    """Addestra una Q-network con Deep Q-Learning e replay buffer.

    Note:
        Richiamata in 01_dql_cartpole e 02_dql_lunarlander (Esercizio 3.2).
    """
    q_network.train()
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    action_dim = int(env.action_space.n)
    global_step = 0

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    losses: list[float] = []
    epsilons: list[float] = []
    eval_episodes: list[int] = []
    eval_mean_rewards: list[float] = []
    eval_mean_lengths: list[float] = []
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_q_state: dict[str, torch.Tensor] | None = None
    best_target_state: dict[str, torch.Tensor] | None = None

    for episode_idx in range(1, num_episodes + 1):
        observation, _ = env.reset()
        total_reward = 0.0
        episode_length = 0

        for _ in range(max_steps):
            epsilon = compute_epsilon(
                step=global_step,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay_steps=epsilon_decay_steps,
            )
            action = select_dqn_action(
                q_network=q_network,
                observation=observation,
                device=device,
                action_dim=action_dim,
                epsilon=epsilon,
            )
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(
                observation=observation,
                action=action,
                reward=float(reward),
                next_observation=next_observation,
                done=done,
            )

            observation = next_observation
            total_reward += float(reward)
            episode_length += 1
            global_step += 1
            epsilons.append(epsilon)

            should_train = (
                len(replay_buffer) >= min_replay_size
                and len(replay_buffer) >= batch_size
                and global_step % train_every_steps == 0
            )
            if should_train:
                loss_value = update_dqn_network(
                    q_network=q_network,
                    target_network=target_network,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    device=device,
                    gamma=gamma,
                    gradient_clip=gradient_clip,
                )
                losses.append(loss_value)

            if (
                target_update_every_steps > 0
                and global_step % target_update_every_steps == 0
            ):
                target_network.load_state_dict(q_network.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

        should_evaluate = (
            eval_env is not None
            and eval_every_n is not None
            and episode_idx % eval_every_n == 0
        )
        if should_evaluate:
            eval_res = evaluate_q_policy(
                env=eval_env,
                q_network=q_network,
                device=device,
                num_episodes=eval_episodes_m,
                max_steps=max_steps,
            )
            eval_episodes.append(episode_idx)
            eval_mean_rewards.append(eval_res["mean_reward"])
            eval_mean_lengths.append(eval_res["mean_length"])
            if best_eval_mean_reward is None or eval_res["mean_reward"] > best_eval_mean_reward:
                from .utils import clone_model_state

                best_eval_episode = episode_idx
                best_eval_mean_reward = eval_res["mean_reward"]
                best_q_state = clone_model_state(q_network)
                best_target_state = clone_model_state(target_network)

    return DQNTrainingResult(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        losses=losses,
        epsilons=epsilons,
        eval_episodes=eval_episodes if eval_every_n is not None else None,
        eval_mean_rewards=eval_mean_rewards if eval_every_n is not None else None,
        eval_mean_lengths=eval_mean_lengths if eval_every_n is not None else None,
        best_eval_episode=best_eval_episode,
        best_eval_mean_reward=best_eval_mean_reward,
        best_q_state=best_q_state,
        best_target_state=best_target_state,
    )


def train_vectorized_dqn(
    envs: gym.vector.VectorEnv,
    q_network: nn.Module,
    target_network: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_episodes: int = 300,
    gamma: float = 0.99,
    max_steps: int = 500,
    batch_size: int = 64,
    replay_capacity: int = 50_000,
    min_replay_size: int = 1_000,
    train_every_steps: int = 1,
    gradient_steps: int = 1,
    target_update_every_steps: int = 500,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 10_000,
    gradient_clip: float | None = 10.0,
    eval_env: gym.Env | None = None,
    eval_every_n: int | None = None,
    eval_episodes_m: int = 10,
) -> DQNTrainingResult:
    """Addestra DQN raccogliendo transizioni da piu' ambienti in parallelo.

    Note:
        Richiamata in 01_dql_cartpole e 02_dql_lunarlander (Esercizio 3.2).
    """
    q_network.train()
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    action_dim = int(envs.single_action_space.n)
    num_envs = int(envs.num_envs)
    global_step = 0
    last_target_update_step = 0

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    losses: list[float] = []
    epsilons: list[float] = []
    eval_episodes: list[int] = []
    eval_mean_rewards: list[float] = []
    eval_mean_lengths: list[float] = []
    best_eval_episode: int | None = None
    best_eval_mean_reward: float | None = None
    best_q_state: dict[str, torch.Tensor] | None = None
    best_target_state: dict[str, torch.Tensor] | None = None

    observations, _ = envs.reset()
    current_episode_rewards = np.zeros(num_envs, dtype=np.float32)
    current_episode_lengths = np.zeros(num_envs, dtype=np.int32)
    completed_episodes = 0

    while completed_episodes < num_episodes:
        epsilon = compute_epsilon(
            step=global_step,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        )
        actions = select_dqn_actions(
            q_network=q_network,
            observations=observations,
            device=device,
            action_dim=action_dim,
            epsilon=epsilon,
        )
        next_observations, rewards, terminated, truncated, _ = envs.step(actions)
        dones = np.logical_or(terminated, truncated)

        for env_idx in range(num_envs):
            replay_buffer.add(
                observation=observations[env_idx],
                action=int(actions[env_idx]),
                reward=float(rewards[env_idx]),
                next_observation=next_observations[env_idx],
                done=bool(dones[env_idx]),
            )

        current_episode_rewards += rewards.astype(np.float32)
        current_episode_lengths += 1
        observations = next_observations
        global_step += num_envs
        epsilons.extend([epsilon] * num_envs)

        should_train = (
            len(replay_buffer) >= min_replay_size
            and len(replay_buffer) >= batch_size
            and (train_every_steps <= 1 or global_step % train_every_steps == 0)
        )
        if should_train:
            for _ in range(max(1, gradient_steps)):
                loss_value = update_dqn_network(
                    q_network=q_network,
                    target_network=target_network,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                    gradient_clip=gradient_clip,
                )
                losses.append(loss_value)

        if (
            target_update_every_steps > 0
            and global_step - last_target_update_step >= target_update_every_steps
        ):
            target_network.load_state_dict(q_network.state_dict())
            last_target_update_step = global_step

        for env_idx, done in enumerate(dones):
            if done and completed_episodes < num_episodes:
                completed_episodes += 1
                episode_rewards.append(float(current_episode_rewards[env_idx]))
                episode_lengths.append(int(current_episode_lengths[env_idx]))
                current_episode_rewards[env_idx] = 0.0
                current_episode_lengths[env_idx] = 0

                should_evaluate = (
                    eval_env is not None
                    and eval_every_n is not None
                    and completed_episodes % eval_every_n == 0
                )
                if should_evaluate:
                    eval_res = evaluate_q_policy(
                        env=eval_env,
                        q_network=q_network,
                        device=device,
                        num_episodes=eval_episodes_m,
                        max_steps=max_steps,
                    )
                    eval_episodes.append(completed_episodes)
                    eval_mean_rewards.append(eval_res["mean_reward"])
                    eval_mean_lengths.append(eval_res["mean_length"])
                    if best_eval_mean_reward is None or eval_res["mean_reward"] > best_eval_mean_reward:
                        from .utils import clone_model_state

                        best_eval_episode = completed_episodes
                        best_eval_mean_reward = eval_res["mean_reward"]
                        best_q_state = clone_model_state(q_network)
                        best_target_state = clone_model_state(target_network)

    return DQNTrainingResult(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        losses=losses,
        epsilons=epsilons,
        eval_episodes=eval_episodes if eval_every_n is not None else None,
        eval_mean_rewards=eval_mean_rewards if eval_every_n is not None else None,
        eval_mean_lengths=eval_mean_lengths if eval_every_n is not None else None,
        best_eval_episode=best_eval_episode,
        best_eval_mean_reward=best_eval_mean_reward,
        best_q_state=best_q_state,
        best_target_state=best_target_state,
    )


def evaluate_q_policy(
    env: gym.Env,
    q_network: nn.Module,
    device: torch.device,
    num_episodes: int = 10,
    max_steps: int = 500,
) -> dict[str, float | list[float] | list[int]]:
    """Valuta una Q-policy scegliendo sempre l'azione con valore Q massimo.

    Note:
        Richiamata in 01_dql_cartpole e 02_dql_lunarlander (Esercizio 3.2).
    """
    q_network.eval()
    rewards: list[float] = []
    lengths: list[int] = []
    action_dim = int(env.action_space.n)

    for _ in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0.0
        episode_length = 0

        for _ in range(max_steps):
            action = select_dqn_action(
                q_network=q_network,
                observation=observation,
                device=device,
                action_dim=action_dim,
                epsilon=0.0,
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


def render_q_policy(
    environment_name: str,
    q_network: nn.Module,
    device: torch.device,
    num_episodes: int = 3,
    max_steps: int = 500,
    frame_delay: float = 0.02,
) -> list[float]:
    """Mostra una Q-policy in tempo reale usando il renderer grafico di Gymnasium.

    Note:
        Richiamata in 01_dql_cartpole e 02_dql_lunarlander (Esercizio 3.2).
    """
    env = gym.make(environment_name, render_mode="human")
    q_network.eval()
    rewards: list[float] = []
    action_dim = int(env.action_space.n)

    try:
        for _ in range(num_episodes):
            observation, _ = env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                action = select_dqn_action(
                    q_network=q_network,
                    observation=observation,
                    device=device,
                    action_dim=action_dim,
                    epsilon=0.0,
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
