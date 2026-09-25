import numpy as np
import pandas as pd

from datathon.data import ARMS, clean
from datathon.reward_model import FEATURES, fit_reward_model, predict_all_arms


def _base_sintetica(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "job": rng.choice(["admin.", "student", "retired"], n),
        "marital": rng.choice(["married", "single"], n),
        "education": rng.choice(["basic.4y", "university.degree"], n),
        "default": rng.choice(["no", "unknown"], n),
        "housing": rng.choice(["no", "yes"], n),
        "loan": rng.choice(["no", "yes"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "month": rng.choice(["may", "jun"], n),
        "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], n),
        "campaign": rng.integers(1, 20, n),
        "pdays": rng.choice([999, 3, 6], n),
        "previous": rng.integers(0, 3, n),
        "poutcome": rng.choice(["nonexistent", "success", "failure"], n),
        "emp.var.rate": rng.normal(0, 1, n),
        "cons.price.idx": rng.normal(93, 0.5, n),
        "cons.conf.idx": rng.normal(-40, 4, n),
        "euribor3m": rng.uniform(0.6, 5, n),
        "nr.employed": rng.normal(5100, 70, n),
        "y": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
    })
    return clean(df)


def test_arms_tem_10_combinacoes_unicas():
    assert len(ARMS) == 10 and len(set(ARMS)) == 10


def test_fit_e_predict_all_arms_tem_shape_correto():
    df = _base_sintetica()
    model = fit_reward_model(df, max_iter=20)
    P = predict_all_arms(model, df.head(7))
    assert P.shape == (7, len(ARMS))
    assert np.all((P >= 0) & (P <= 1))


def test_features_nao_incluem_recompensa_nem_alvo():
    assert "reward" not in FEATURES and "y" not in FEATURES
