"""Carga e preparação da base Telemarketing JYB (Kaggle: aguado/telemarketing-jyb-dataset).

Este módulo é a única fonte de verdade para ler e limpar os dados.
O notebook de EDA e a API importam daqui, garantindo que o modelo
avaliado e o modelo servido veem exatamente a mesma transformação.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Código usado na base UCI para "cliente nunca contatado em campanha anterior"
PDAYS_NUNCA_CONTATADO = 999

# Corte da cauda longa de `campaign` (percentil 99 no treino)
CAMPAIGN_MAX = 14

# Variáveis que o banco CONTROLA antes de ligar -> viram os braços do bandit
ARM_COLS = ["contact", "day_of_week"]

# Os 10 braços: canal x dia da semana (ordem fixa, usada em toda a matriz de recompensas)
ARMS = [f"{c}_{d}" for c in ["cellular", "telephone"] for d in ["mon", "tue", "wed", "thu", "fri"]]

# Colunas de contexto (o bandit observa, mas não escolhe)
CONTEXT_NUM = [
    "age", "campaign", "pdays", "previous",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
    "contato_previo",
]
CONTEXT_CAT = [
    "job", "marital", "education", "default", "housing", "loan", "poutcome", "month",
]


def load_raw(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Lê `train.csv` ou `test.csv` tratando as particularidades do arquivo.

    - separador `;`
    - `Unnamed: 0` é o índice da base UCI original e é descartado
    """
    df = pd.read_csv(raw_dir / f"{name}.csv", sep=";")
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as decisões de limpeza documentadas no notebook 01_eda.

    1. Renomeia colunas com ponto (`emp.var.rate` -> `emp_var_rate`).
    2. `pdays == 999` vira flag `contato_previo = 0` e o valor passa a ausente (NaN),
       porque 999 é um código, não uma contagem de dias.
    3. `campaign` é limitado a CAMPAIGN_MAX para conter a cauda longa.
    4. Cria `arm` = canal + dia da semana (ação do bandit), quando ambos existirem.
       Um cliente novo (API) não tem canal nem dia: são o que a política decide.
    5. Cria `reward` = 1 se `y == "yes"`, quando `y` existir.

    Mantém deliberadamente: linhas duplicadas (sem id de cliente, cada linha
    é um contato real) e a categoria `unknown` (não saber é informação).
    """
    out = df.copy()
    out.columns = [c.replace(".", "_") for c in out.columns]

    out["contato_previo"] = (out["pdays"] != PDAYS_NUNCA_CONTATADO).astype(int)
    out["pdays"] = out["pdays"].where(out["contato_previo"] == 1, np.nan)

    out["campaign"] = out["campaign"].clip(upper=CAMPAIGN_MAX)

    if all(c in out.columns for c in ARM_COLS):
        out["arm"] = out["contact"] + "_" + out["day_of_week"]

    if "y" in out.columns:
        out["reward"] = (out["y"] == "yes").astype(int)
        out = out.drop(columns=["y"])

    return out


def load_processed(name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Lê a base já tratada, salva pelo notebook 01_eda em Parquet."""
    return pd.read_parquet(processed_dir / f"{name}.parquet")
