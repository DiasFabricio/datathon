"""Recomendação para um cliente: a peça compartilhada pelo Golden Set (Etapa 4)
e pela API (Etapa 5).

Duas fontes entram na resposta:
- A POLÍTICA (Thompson Sampling treinado na simulação) decide o braço, isto é,
  por qual canal e em que dia ligar. Ela sorteia uma taxa de cada crença Beta e
  escolhe o maior sorteio; por isso a decisão pode variar entre chamadas, e o
  campo `explore` indica quando o braço sorteado não é o de maior média.
- O MODELO DE RECOMPENSA (mesmo do simulador) usa o contexto do cliente para
  estimar a propensão de conversão no braço recomendado. Não muda a decisão do
  braço; serve para priorizar a ligação e explicar a resposta.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from datathon.bandits import ThompsonSamplingBeta
from datathon.data import ARMS, CONTEXT_CAT, CONTEXT_NUM, PROCESSED_DIR, clean
from datathon.reward_model import predict_all_arms

MODELS_DIR = PROCESSED_DIR.parent.parent / "models"
POLICY_PATH = MODELS_DIR / "policy_thompson.json"
REWARD_MODEL_PATH = MODELS_DIR / "reward_model.joblib"

# Campos que um cliente novo informa (contato_previo é derivado de pdays)
CLIENT_FIELDS = [c for c in CONTEXT_NUM if c != "contato_previo"] + CONTEXT_CAT


def load_artifacts(models_dir: Path = MODELS_DIR, seed: int | None = None):
    policy = ThompsonSamplingBeta.load(models_dir / POLICY_PATH.name, seed=seed if seed is not None else 0)
    reward_model = joblib.load(models_dir / REWARD_MODEL_PATH.name)
    return policy, reward_model


def client_to_frame(client: dict) -> pd.DataFrame:
    """Aceita nomes com ponto (`emp.var.rate`) ou underscore e devolve a linha limpa."""
    row = {k.replace("_", ".") if k in {"emp_var_rate", "cons_price_idx", "cons_conf_idx", "nr_employed"} else k: v
           for k, v in client.items()}
    df = pd.DataFrame([row])
    faltando = [c for c in CLIENT_FIELDS if c.replace("_", ".") not in df.columns and c not in df.columns]
    if faltando:
        raise ValueError(f"campos ausentes: {faltando}")
    return clean(df)


def recommend(client: dict, policy: ThompsonSamplingBeta, reward_model, sample: bool = True) -> dict:
    """Devolve a decisão para um cliente.

    sample=True  -> Thompson Sampling de verdade (sorteia; pode explorar).
    sample=False -> braço de maior média posterior (determinístico; útil em testes).
    """
    x = client_to_frame(client)
    arm_idx = policy.select() if sample else int(np.argmax(policy.means))
    arm = ARMS[arm_idx]
    contact, day = arm.split("_")

    probs = predict_all_arms(reward_model, x)[0]          # propensão do cliente em cada braço
    ranking = sorted(zip(ARMS, probs), key=lambda t: -t[1])

    return {
        "arm": arm,
        "contact": contact,
        "day_of_week": day,
        "propensity": float(probs[arm_idx]),
        "policy_mean": float(policy.means[arm_idx]),
        "explore": bool(arm_idx != int(np.argmax(policy.means))),
        "best_arm_for_client": ranking[0][0],
        "best_arm_propensity": float(ranking[0][1]),
    }
