import numpy as np
import pandas as pd

from datathon.data import CAMPAIGN_MAX, clean


def _amostra() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [30, 45],
        "contact": ["cellular", "telephone"],
        "day_of_week": ["mon", "fri"],
        "pdays": [999, 6],
        "campaign": [1, 40],
        "emp.var.rate": [1.1, -1.8],
        "y": ["no", "yes"],
    })


def test_clean_renomeia_colunas_com_ponto():
    out = clean(_amostra())
    assert "emp_var_rate" in out.columns
    assert "emp.var.rate" not in out.columns


def test_clean_trata_pdays_999_como_nunca_contatado():
    out = clean(_amostra())
    assert out["contato_previo"].tolist() == [0, 1]
    assert np.isnan(out.loc[0, "pdays"])
    assert out.loc[1, "pdays"] == 6


def test_clean_limita_campaign():
    out = clean(_amostra())
    assert out["campaign"].max() == CAMPAIGN_MAX


def test_clean_cria_arm_e_reward():
    out = clean(_amostra())
    assert out["arm"].tolist() == ["cellular_mon", "telephone_fri"]
    assert out["reward"].tolist() == [0, 1]
    assert "y" not in out.columns


def test_clean_sem_y_nao_cria_reward():
    out = clean(_amostra().drop(columns=["y"]))
    assert "reward" not in out.columns
