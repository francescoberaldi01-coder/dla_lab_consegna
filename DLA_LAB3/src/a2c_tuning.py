"""Supporto sperimentale per la ricerca di iperparametri A2C su LunarLander."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from torch.optim import Adam

from .a2c import A2CTrainingResult, EvalCallback, train_a2c, train_vectorized_a2c
from .evaluation import evaluate_policy
from .networks import PolicyNetwork, ValueNetwork
from .utils import make_vector_env, restore_model_state, seed_environment, set_seed


@dataclass
class A2CExperimentOutput:
    """Risultato completo di una run A2C usata per baseline o conferma HPO."""

    seed: int
    config: dict[str, Any]
    training_result: A2CTrainingResult
    final_evaluation: dict[str, float | list[float] | list[int]]
    summary: dict[str, float | int | None]
    actor: PolicyNetwork
    critic: ValueNetwork


def suggest_a2c_lunarlander_params(trial: Any) -> dict[str, Any]:
    """Campiona uno spazio di ricerca compatto e interpretabile per A2C LunarLander."""
    num_envs = trial.suggest_categorical("num_envs", [1, 4, 8])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "rollout_steps": trial.suggest_categorical("rollout_steps", [5, 10, 20]),
        "value_loss_weight": trial.suggest_float("value_loss_weight", 0.25, 1.0),
        "entropy_weight": trial.suggest_float("entropy_weight", 0.0, 0.01),
        "standardize_advantages": True,
        "vectorized": num_envs > 1,
        "num_envs": num_envs,
        "vector_env_type": "sync",
    }


def build_a2c_lunarlander_config(
    base_config: dict[str, Any],
    trial_params: dict[str, Any],
    num_episodes: int | None = None,
    eval_every_n: int | None = None,
    eval_episodes_m: int | None = None,
) -> dict[str, Any]:
    """Crea una configurazione A2C completa applicando i parametri del trial.

    Note:
        Richiamata in 03_a2c_lunarlander_hyperparameter_search (Esercizio 3.1).
    """
    config = deepcopy(base_config)
    config["training"].update(trial_params)

    if "num_envs" in trial_params and "vectorized" not in trial_params:
        config["training"]["vectorized"] = int(trial_params["num_envs"]) > 1
    if "num_envs" in trial_params and "vector_env_type" not in trial_params:
        config["training"]["vector_env_type"] = "sync"
    if "standardize_advantages" not in trial_params:
        config["training"]["standardize_advantages"] = True

    if num_episodes is not None:
        config["training"]["num_episodes"] = num_episodes
    if eval_every_n is not None:
        config["evaluation"]["eval_every_n"] = eval_every_n
    if eval_episodes_m is not None:
        config["evaluation"]["eval_episodes_m"] = eval_episodes_m

    return config


def _get_space_dimensions(env: gym.Env | gym.vector.VectorEnv) -> tuple[int, int]:
    """Legge dimensione osservazioni e numero azioni da env singolo o vettoriale."""
    observation_space = getattr(env, "single_observation_space", env.observation_space)
    action_space = getattr(env, "single_action_space", env.action_space)
    return int(observation_space.shape[0]), int(action_space.n)


def _make_training_env(config: dict[str, Any], seed: int) -> gym.Env | gym.vector.VectorEnv:
    """Crea l'ambiente di training coerente con la configurazione."""
    training_config = config["training"]
    environment_name = config["environment"]["name"]
    use_vectorized = (
        bool(training_config.get("vectorized", False))
        and int(training_config.get("num_envs", 1)) > 1
    )

    if use_vectorized:
        return make_vector_env(
            environment_name=environment_name,
            seed=seed,
            num_envs=int(training_config["num_envs"]),
            vector_env_type=training_config.get("vector_env_type", "sync"),
        )

    env = gym.make(environment_name)
    seed_environment(env, seed)
    return env


def run_a2c_lunarlander_config(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    eval_callback: EvalCallback | None = None,
) -> A2CExperimentOutput:
    """Esegue una run A2C LunarLander completa e valuta il miglior checkpoint.

    Note:
        Richiamata in 03_a2c_lunarlander_hyperparameter_search (Esercizio 3.1).
    """
    set_seed(seed)
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    train_env = None
    eval_env = None
    final_eval_env = None

    try:
        train_env = _make_training_env(config=config, seed=seed)
        observation_dim, action_dim = _get_space_dimensions(train_env)
        actor = PolicyNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=int(training_config["hidden_dim"]),
        ).to(device)
        critic = ValueNetwork(
            observation_dim=observation_dim,
            hidden_dim=int(training_config["hidden_dim"]),
        ).to(device)
        optimizer = Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=training_config["learning_rate"],
        )
    except Exception:
        if train_env is not None:
            train_env.close()
        raise

    try:
        eval_env = gym.make(config["environment"]["name"])
        seed_environment(eval_env, seed + 10_000)
        use_vectorized = (
            bool(training_config.get("vectorized", False))
            and int(training_config.get("num_envs", 1)) > 1
        )

        common_kwargs = {
            "actor": actor,
            "critic": critic,
            "optimizer": optimizer,
            "device": device,
            "num_episodes": int(training_config["num_episodes"]),
            "gamma": float(training_config["gamma"]),
            "max_steps": int(config["environment"]["max_steps"]),
            "rollout_steps": int(training_config["rollout_steps"]),
            "value_loss_weight": float(training_config["value_loss_weight"]),
            "standardize_advantages": bool(training_config["standardize_advantages"]),
            "standardize_critic_targets": bool(training_config.get("standardize_critic_targets", False)),
            "entropy_weight": float(training_config["entropy_weight"]),
            "eval_env": eval_env,
            "eval_every_n": evaluation_config["eval_every_n"],
            "eval_episodes_m": evaluation_config["eval_episodes_m"],
            "eval_callback": eval_callback,
        }
        if use_vectorized:
            training_result = train_vectorized_a2c(envs=train_env, **common_kwargs)
        else:
            training_result = train_a2c(env=train_env, **common_kwargs)
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()

    restore_model_state(actor, training_result.best_actor_state)
    restore_model_state(critic, training_result.best_critic_state)

    try:
        final_eval_env = gym.make(config["environment"]["name"])
        seed_environment(final_eval_env, seed + 20_000)
        final_evaluation = evaluate_policy(
            env=final_eval_env,
            policy=actor,
            device=device,
            num_episodes=int(evaluation_config["num_episodes"]),
            max_steps=int(config["environment"]["max_steps"]),
            deterministic=bool(evaluation_config.get("deterministic", True)),
        )
    finally:
        if final_eval_env is not None:
            final_eval_env.close()

    best_reward = training_result.best_eval_mean_reward
    summary = {
        "Seed": seed,
        "Final Train Reward": round(training_result.episode_rewards[-1], 1),
        "Best Train Reward": round(max(training_result.episode_rewards), 1),
        "Eval Mean Reward": round(final_evaluation["mean_reward"], 1),
        "Eval Min Reward": round(final_evaluation["min_reward"], 1),
        "Eval Max Reward": round(final_evaluation["max_reward"], 1),
        "Eval Mean Length": round(final_evaluation["mean_length"], 1),
        "Best Checkpoint Episode": training_result.best_eval_episode,
        "Best Checkpoint Reward": round(best_reward, 1) if best_reward is not None else None,
    }

    return A2CExperimentOutput(
        seed=seed,
        config=config,
        training_result=training_result,
        final_evaluation=final_evaluation,
        summary=summary,
        actor=actor,
        critic=critic,
    )


def make_optuna_eval_callback(trial: Any) -> EvalCallback:
    """Crea un callback che comunica a Optuna le valutazioni periodiche."""
    def _callback(episode: int, eval_result: dict[str, float | list[float] | list[int]]) -> None:
        trial.report(float(eval_result["mean_reward"]), step=episode)
        if trial.should_prune():
            import optuna

            raise optuna.TrialPruned()

    return _callback


def create_a2c_lunarlander_objective(
    base_config: dict[str, Any],
    device: torch.device,
    seed: int = 42,
    trial_num_episodes: int = 600,
    trial_eval_every_n: int = 50,
    trial_eval_episodes_m: int = 5,
):
    """Restituisce una objective Optuna con budget ridotto per ogni trial."""
    def _objective(trial: Any) -> float:
        trial_params = suggest_a2c_lunarlander_params(trial)
        trial_config = build_a2c_lunarlander_config(
            base_config=base_config,
            trial_params=trial_params,
            num_episodes=trial_num_episodes,
            eval_every_n=trial_eval_every_n,
            eval_episodes_m=trial_eval_episodes_m,
        )
        output = run_a2c_lunarlander_config(
            config=trial_config,
            seed=seed + trial.number * 1000,
            device=device,
            eval_callback=make_optuna_eval_callback(trial),
        )
        best_reward = output.training_result.best_eval_mean_reward
        objective_value = (
            float(best_reward)
            if best_reward is not None
            else float(output.final_evaluation["mean_reward"])
        )

        trial.set_user_attr("best_eval_episode", output.training_result.best_eval_episode)
        trial.set_user_attr("final_eval_mean_reward", output.final_evaluation["mean_reward"])
        trial.set_user_attr("final_eval_min_reward", output.final_evaluation["min_reward"])
        trial.set_user_attr("final_eval_max_reward", output.final_evaluation["max_reward"])
        return objective_value

    return _objective


def run_a2c_lunarlander_study(
    base_config: dict[str, Any],
    device: torch.device,
    n_trials: int = 20,
    seed: int = 42,
    trial_num_episodes: int = 600,
    trial_eval_every_n: int = 50,
    trial_eval_episodes_m: int = 5,
    n_jobs: int = 1,
):
    """Lancia uno study Optuna per A2C LunarLander con pruning mediano.

    Note:
        Richiamata in 03_a2c_lunarlander_hyperparameter_search (Esercizio 3.1).
    """
    import optuna

    if n_jobs != 1:
        # Con più trial paralleli, il thread pool interno di PyTorch per ogni trial
        # andrebbe in oversubscription rispetto ai thread di Optuna: lo limitiamo a 1.
        torch.set_num_threads(1)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=trial_eval_every_n * 2,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="a2c_lunarlander_hpo",
    )
    objective = create_a2c_lunarlander_objective(
        base_config=base_config,
        device=device,
        seed=seed,
        trial_num_episodes=trial_num_episodes,
        trial_eval_every_n=trial_eval_every_n,
        trial_eval_episodes_m=trial_eval_episodes_m,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study
