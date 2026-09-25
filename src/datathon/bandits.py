"""Políticas de decisão: baseline determinístico e multi-armed bandits.

Interface comum (duck typing):
    select(x) -> int        índice do braço escolhido (x = contexto ou None)
    update(x, arm, reward)  aprende com a recompensa observada

As políticas sem contexto ignoram `x`. Assim, o mesmo laço de simulação
serve para todas, e a API da Etapa 5 chama `select` da mesma forma.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FixedArm:
    """Baseline determinístico: escolhe sempre o mesmo braço e não aprende.

    Representa a regra fixa do enunciado ("oferecer sempre o melhor
    histórico"). Não tem exploração nenhuma.
    """

    name = "fixed"

    def __init__(self, arm: int):
        self.arm = arm

    def select(self, x=None) -> int:
        return self.arm

    def update(self, x, arm: int, reward: int) -> None:
        pass


class RandomPolicy:
    """Piso de referência: escolhe um braço ao acaso (exploração pura)."""

    name = "random"

    def __init__(self, n_arms: int, seed: int = 0):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

    def select(self, x=None) -> int:
        return int(self.rng.integers(self.n_arms))

    def update(self, x, arm: int, reward: int) -> None:
        pass


class EpsilonGreedy:
    """Com probabilidade epsilon explora (braço aleatório); senão explota o
    braço de maior taxa média observada.

    É o bandit mais simples. O trade-off é explícito no parâmetro: epsilon
    alto aprende mais rápido mas desperdiça tráfego para sempre; epsilon
    baixo pode travar cedo em um braço mediano.
    """

    name = "epsilon_greedy"

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 0):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms, dtype=int)
        self.sums = np.zeros(n_arms, dtype=float)

    @property
    def means(self) -> np.ndarray:
        # braços nunca puxados recebem média otimista (1.0) para serem tentados uma vez
        return np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 1.0)

    def select(self, x=None) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return int(self.rng.choice(np.flatnonzero(self.means == self.means.max())))

    def update(self, x, arm: int, reward: int) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


class ThompsonSamplingBeta:
    """Thompson Sampling para recompensas binárias (Bernoulli).

    Cada braço tem uma crença Beta(alpha, beta) sobre sua taxa de conversão.
    A cada decisão, sorteamos uma taxa de cada crença e escolhemos o braço
    com o maior sorteio. Braços pouco testados têm crenças largas e por isso
    às vezes "ganham o sorteio" (exploração); braços bem conhecidos e bons
    ganham quase sempre (explotação). A exploração diminui sozinha conforme
    as crenças se estreitam: não há epsilon para ajustar.

    Prior: Beta(alpha0, beta0). Com (1, 1) é uniforme, "não sei nada".
    Após observar s sucessos e f falhas, a posterior é Beta(alpha0+s, beta0+f).
    """

    name = "thompson_beta"

    def __init__(self, n_arms: int, alpha0: float = 1.0, beta0: float = 1.0, seed: int = 0):
        self.n_arms = n_arms
        self.alpha0, self.beta0 = alpha0, beta0
        self.rng = np.random.default_rng(seed)
        self.successes = np.zeros(n_arms, dtype=float)
        self.failures = np.zeros(n_arms, dtype=float)

    @property
    def alpha(self) -> np.ndarray:
        return self.alpha0 + self.successes

    @property
    def beta(self) -> np.ndarray:
        return self.beta0 + self.failures

    @property
    def means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def select(self, x=None) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, x, arm: int, reward: int) -> None:
        if reward:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

    # --- persistência: o "modelo" do bandit são suas contagens ---
    def to_dict(self) -> dict:
        return {
            "alpha0": self.alpha0, "beta0": self.beta0,
            "successes": self.successes.tolist(), "failures": self.failures.tolist(),
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path, seed: int = 0) -> "ThompsonSamplingBeta":
        d = json.loads(Path(path).read_text())
        obj = cls(len(d["successes"]), d["alpha0"], d["beta0"], seed=seed)
        obj.successes[:] = d["successes"]
        obj.failures[:] = d["failures"]
        return obj
