"""Utility di plotting per visualizzare gli esperimenti del Lab 3."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_training_curve(rewards: list[float], window: int | None = None) -> None:
    """Disegna i reward episodici senza applicare smoothing."""
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, linewidth=1.5, label="Reward per episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Andamento del training")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()


def plot_seed_training_curves(results: dict[int, list[float]]) -> None:
    """Disegna una curva di training separata per ogni seed sperimentale."""
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.6 * len(results)), sharex=True, sharey=True)

    if len(results) == 1:
        axes = [axes]

    for axis, (seed, rewards) in zip(axes, results.items()):
        axis.plot(rewards, linewidth=1.4)
        axis.set_title(f"Seed {seed}")
        axis.set_ylabel("Reward")
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Episodio")
    fig.suptitle("Confronto tra seed - reward per episodio", y=0.995)
    fig.tight_layout()
    plt.show()


def plot_comparison_curves(results: dict[str, list[float]], window: int | None = None) -> None:
    """Disegna piu' curve di reward episodico senza applicare smoothing."""
    plt.figure(figsize=(10, 4))

    for label, rewards in results.items():
        plt.plot(rewards, linewidth=1.5, label=label)

    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Confronto tra varianti di REINFORCE")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()


def plot_seed_eval_curves(results: dict[int, dict[str, list]]) -> None:
    """Disegna sia le curve di training che di valutazione periodica per ogni seed.

    Note:
        Richiamata in 01_reinforce_cartpole (Esercizio 1); in 01_a2c_cartpole e
        02_a2c_lunarlander (Esercizio 3.1); in 01_dql_cartpole e
        02_dql_lunarlander (Esercizio 3.2).
    """
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 3.0 * len(results)), sharex=True, sharey=True)

    if len(results) == 1:
        axes = [axes]

    for axis, (seed, data) in zip(axes, results.items()):
        train_rewards = data["train_rewards"]
        eval_intervals = data["eval_intervals"]
        eval_rewards = data["eval_rewards"]

        axis.plot(train_rewards, alpha=0.35, label="Train (stocastico)", color="gray")
        axis.plot(eval_intervals, eval_rewards, marker="o", markersize=4, linewidth=1.8, label="Val (deterministica)", color="teal")
        axis.set_title(f"Seed {seed}")
        axis.set_ylabel("Reward")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")

    axes[-1].set_xlabel("Episodio")
    fig.suptitle("Confronto tra seed - Training vs Valutazione Periodica", y=0.995)
    fig.tight_layout()
    plt.show()


def plot_eval_comparison_by_seed(results: dict[int, dict[str, dict[str, list]]]) -> None:
    """Disegna le curve di valutazione periodica di REINFORCE Standard e con Baseline per ogni seed."""
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 3.0 * len(results)), sharex=True, sharey=True)

    if len(results) == 1:
        axes = [axes]

    for axis, (seed, data) in zip(axes, results.items()):
        std_intervals = data["Standard"]["eval_intervals"]
        std_rewards = data["Standard"]["eval_rewards"]
        base_intervals = data["Baseline"]["eval_intervals"]
        base_rewards = data["Baseline"]["eval_rewards"]

        axis.plot(std_intervals, std_rewards, marker="o", markersize=4, linewidth=1.8, label="REINFORCE Standard", color="#E67E22")
        axis.plot(base_intervals, base_rewards, marker="s", markersize=4, linewidth=1.8, label="REINFORCE + Baseline", color="#2E86C1")
        axis.set_title(f"Seed {seed}")
        axis.set_ylabel("Reward Val.")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")

    axes[-1].set_xlabel("Episodio")
    fig.suptitle("Confronto Valutazione Deterministica - Standard vs Value Baseline", y=0.995)
    fig.tight_layout()
    plt.show()

