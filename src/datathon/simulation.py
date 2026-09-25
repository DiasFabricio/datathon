"""Simulação offline de políticas contra a matriz de recompensas P.

P[i, a] = P(conversão | cliente i, braço a), estimada pelo modelo de
recompensa. A cada passo a política escolhe um braço e a recompensa é
sorteada com essa probabilidade.

Duas escolhas de método que tornam a comparação justa:
- Números aleatórios comuns: para uma mesma semente, todas as políticas
  veem os mesmos clientes, na mesma ordem, e o mesmo sorteio uniforme U.
  Assim a diferença entre políticas vem da decisão, não da sorte.
- Várias sementes: reportamos média e desvio, porque um único run pode
  enganar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    arms: np.ndarray        # braço escolhido em cada passo
    rewards: np.ndarray     # recompensa sorteada (0/1) em cada passo
    regret: np.ndarray      # arrependimento esperado acumulado vs. oráculo
    order: np.ndarray       # índice do cliente em cada passo

    @property
    def conversion_rate(self) -> float:
        return float(self.rewards.mean())


def simulate(policy, P: np.ndarray, X: np.ndarray | None = None, seed: int = 0) -> SimResult:
    rng = np.random.default_rng(seed)
    n, _ = P.shape
    order = rng.permutation(n)
    U = rng.random(n)

    arms = np.empty(n, dtype=int)
    rewards = np.empty(n, dtype=int)
    for t, i in enumerate(order):
        x = None if X is None else X[i]
        a = policy.select(x)
        r = int(U[t] < P[i, a])
        policy.update(x, a, r)
        arms[t] = a
        rewards[t] = r

    oracle = P[order].max(axis=1)
    regret = np.cumsum(oracle - P[order, arms])
    return SimResult(arms, rewards, regret, order)


def run_experiment(
    make_policy: Callable[[int], object],
    P: np.ndarray,
    X: np.ndarray | None = None,
    seeds: range = range(10),
) -> list[SimResult]:
    """Roda `make_policy(seed)` para cada semente. A política é recriada a cada
    run para começar sem memória."""
    return [simulate(make_policy(s), P, X, seed=s) for s in seeds]


def summarize(results: dict[str, list[SimResult]]) -> pd.DataFrame:
    """Tabela: conversão média ± desvio e regret final por política."""
    rows = []
    for name, runs in results.items():
        conv = np.array([r.conversion_rate for r in runs])
        reg = np.array([r.regret[-1] for r in runs])
        rows.append({
            "política": name,
            "conversão (%)": 100 * conv.mean(),
            "± desvio": 100 * conv.std(),
            "regret final": reg.mean(),
            "runs": len(runs),
        })
    return pd.DataFrame(rows).set_index("política").sort_values("conversão (%)", ascending=False)
