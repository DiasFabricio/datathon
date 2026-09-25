import numpy as np
import pandas as pd
import pytest

from datathon.bandits import ThompsonSamplingBeta
from datathon.data import ARMS
from datathon.recommender import client_to_frame, recommend
from datathon.reward_model import fit_reward_model
from tests.test_reward_model import _base_sintetica

CLIENTE = {
    "age": 41, "job": "admin.", "marital": "married", "education": "university.degree",
    "default": "no", "housing": "yes", "loan": "no", "month": "may",
    "campaign": 2, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp_var_rate": 1.1, "cons_price_idx": 93.994, "cons_conf_idx": -36.4,
    "euribor3m": 4.857, "nr_employed": 5191.0,
}


def test_client_to_frame_nao_cria_arm_e_deriva_flag():
    x = client_to_frame(CLIENTE)
    assert "arm" not in x.columns
    assert x.loc[0, "contato_previo"] == 0 and np.isnan(x.loc[0, "pdays"])


def test_client_to_frame_reclama_de_campo_ausente():
    with pytest.raises(ValueError, match="campos ausentes"):
        client_to_frame({k: v for k, v in CLIENTE.items() if k != "age"})


def test_recommend_deterministico_devolve_braco_de_maior_media():
    policy = ThompsonSamplingBeta(len(ARMS))
    policy.successes[ARMS.index("cellular_wed")] = 50   # crença forte em um braço
    model = fit_reward_model(_base_sintetica(), max_iter=20)
    rec = recommend(CLIENTE, policy, model, sample=False)
    assert rec["arm"] == "cellular_wed" and rec["explore"] is False
    assert rec["contact"] == "cellular" and rec["day_of_week"] == "wed"
    assert 0 <= rec["propensity"] <= 1


def test_policy_save_load_roundtrip(tmp_path):
    p = ThompsonSamplingBeta(3); p.update(None, 1, 1); p.update(None, 2, 0)
    p.save(tmp_path / "pol.json")
    q = ThompsonSamplingBeta.load(tmp_path / "pol.json")
    assert q.successes.tolist() == [0, 1, 0] and q.failures.tolist() == [0, 0, 1]
