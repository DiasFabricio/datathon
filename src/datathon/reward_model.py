"""Modelo de recompensa: estima P(conversão | contexto do cliente, braço).

Este modelo NÃO é a política. Ele é o "mundo simulado": responde, para qualquer
cliente e qualquer braço, qual a chance de conversão. Isso permite avaliar
políticas de bandit offline mesmo quando a base só registra o braço que o
banco de fato usou.

Escolha do algoritmo: gradient boosting em vez de regressão logística. Ambos
têm AUC ≈ 0,79, mas a logística (sem termos de interação) aponta o mesmo
melhor braço para todos os clientes, o que tornaria um bandit contextual
inútil. O boosting captura interações contexto × braço e faz o melhor braço
variar por perfil, o que corresponde melhor à realidade que queremos simular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datathon.data import ARMS, CONTEXT_CAT, CONTEXT_NUM

FEATURES = CONTEXT_NUM + CONTEXT_CAT + ["arm"]

# Hiperparâmetros modestos para reduzir sobreajuste (registrados no MLflow na Etapa 7)
REWARD_MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "random_state": 42,
}


def build_preprocessor() -> ColumnTransformer:
    """Imputa e padroniza numéricas; one-hot nas categóricas e no braço."""
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler()), CONTEXT_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CONTEXT_CAT + ["arm"]),
    ])


def build_reward_model(**overrides) -> Pipeline:
    params = {**REWARD_MODEL_PARAMS, **overrides}
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", HistGradientBoostingClassifier(**params)),
    ])


def fit_reward_model(df: pd.DataFrame, **overrides) -> Pipeline:
    """Treina o modelo em um DataFrame já tratado por `datathon.data.clean`."""
    return build_reward_model(**overrides).fit(df[FEATURES], df["reward"])


def predict_all_arms(model: Pipeline, contexts: pd.DataFrame, arms: list[str] = ARMS) -> np.ndarray:
    """Matriz (n_clientes, n_braços) com P(conversão) de cada cliente em cada braço.

    É a peça central da simulação: para cada cliente, a política escolhe um
    braço e a recompensa é sorteada com a probabilidade da coluna correspondente.
    """
    base = contexts[[c for c in FEATURES if c != "arm"]]
    cols = [model.predict_proba(base.assign(arm=a))[:, 1] for a in arms]
    return np.column_stack(cols)
