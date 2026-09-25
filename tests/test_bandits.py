import numpy as np

from datathon.bandits import EpsilonGreedy, FixedArm, RandomPolicy, ThompsonSamplingBeta
from datathon.simulation import simulate


def _P_toy(n=3000, seed=0):
    # 3 braços com taxas fixas; o braço 2 é claramente o melhor
    rng = np.random.default_rng(seed)
    return np.tile([0.05, 0.10, 0.30], (n, 1)) + rng.normal(0, 0.005, (n, 3))


def test_fixed_arm_sempre_o_mesmo():
    p = FixedArm(1)
    assert {p.select() for _ in range(50)} == {1}


def test_random_cobre_todos_os_bracos():
    p = RandomPolicy(3, seed=0)
    assert {p.select() for _ in range(200)} == {0, 1, 2}


def test_thompson_converge_para_o_melhor_braco():
    res = simulate(ThompsonSamplingBeta(3, seed=0), _P_toy(), seed=0)
    ultimos = res.arms[-500:]
    assert (ultimos == 2).mean() > 0.9


def test_epsilon_greedy_converge_e_explora_na_taxa_certa():
    res = simulate(EpsilonGreedy(3, epsilon=0.2, seed=0), _P_toy(), seed=0)
    ultimos = res.arms[-1000:]
    frac_melhor = (ultimos == 2).mean()
    # explota o melhor ~ (1 - eps) + eps/3 ≈ 0.87
    assert 0.75 < frac_melhor < 0.95


def test_posterior_beta_atualiza():
    p = ThompsonSamplingBeta(2)
    p.update(None, 0, 1); p.update(None, 0, 0); p.update(None, 0, 0)
    assert p.alpha[0] == 2 and p.beta[0] == 3
    assert abs(p.means[0] - 0.4) < 1e-9


def test_regret_do_oraculo_e_zero():
    P = _P_toy()
    res = simulate(FixedArm(2), P, seed=0)
    # braço 2 é o melhor em quase todas as linhas -> regret ~ 0
    assert res.regret[-1] < 1.0
