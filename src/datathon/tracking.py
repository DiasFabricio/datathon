"""Etapa 7 · Registro de experimentos no MLflow (local).

    uv run python -m datathon.tracking     # registra as execuções em ./mlruns
    uv run mlflow ui                       # http://127.0.0.1:5000

Uma execução (run) por política, com os parâmetros que a definem e as métricas
da Etapa 3, mais uma execução para o modelo de recompensa (simulador). Os
artefatos (política em JSON, métricas) são anexados à execução do Thompson
Sampling, que é a política escolhida para servir.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np

from datathon.bandits import EpsilonGreedy, FixedArm, ThompsonSamplingBeta
from datathon.data import ARMS, load_processed
from datathon.recommender import MODELS_DIR, POLICY_PATH
from datathon.reward_model import REWARD_MODEL_PARAMS
from datathon.simulation import run_experiment

EXPERIMENT = "datathon-bandit-telemarketing"
DATASET = "kaggle:aguado/telemarketing-jyb-dataset (v1)"
SEEDS = range(10)


def _metricas_politica(runs, conv_baseline: float, oraculo: float) -> dict:
    conv = np.array([r.conversion_rate for r in runs])
    return {
        "conversao_media": conv.mean(),
        "conversao_desvio": conv.std(),
        "lift_pp_vs_baseline": 100 * (conv.mean() - conv_baseline),
        "conversoes_extra_por_10k": 10_000 * (conv.mean() - conv_baseline),
        "regret_final_medio": float(np.mean([r.regret[-1] for r in runs])),
        "fracao_ganho_capturada": (conv.mean() - conv_baseline) / (oraculo - conv_baseline),
    }


def main() -> None:
    mlflow.set_experiment(EXPERIMENT)
    df = load_processed("train")
    P = np.load(MODELS_DIR / "P_matrix.npy")
    k = len(ARMS)
    logged = df["arm"].map({a: i for i, a in enumerate(ARMS)}).to_numpy()
    conv_banco = float(P[np.arange(len(P)), logged].mean())
    oraculo = float(P.max(1).mean())
    metrics_json = json.load(open(MODELS_DIR / "metrics.json"))

    comuns = {"dataset": DATASET, "n_clientes": len(df), "n_bracos": k, "n_seeds": len(SEEDS),
              "avaliacao": "simulador HistGradientBoosting + numeros aleatorios comuns"}

    # 1) modelo de recompensa (simulador)
    with mlflow.start_run(run_name="reward_model_hgb"):
        mlflow.set_tags({"tipo": "modelo_recompensa", "dataset": DATASET})
        mlflow.log_params(REWARD_MODEL_PARAMS)
        mlflow.log_metrics(metrics_json["modelo_recompensa"])

    # 2) baseline: política atual do banco (sem simulação de decisão)
    with mlflow.start_run(run_name="baseline_politica_banco"):
        mlflow.set_tags({"tipo": "politica", "familia": "baseline"})
        mlflow.log_params({**comuns, "politica": "braço registrado na base"})
        mlflow.log_metrics({"conversao_media": conv_banco, "lift_pp_vs_baseline": 0.0, "conversao_oraculo": oraculo})

    # 3) políticas adaptativas
    politicas = {
        "epsilon_greedy_0.05": (lambda s: EpsilonGreedy(k, epsilon=0.05, seed=s), {"epsilon": 0.05}),
        "epsilon_greedy_0.10": (lambda s: EpsilonGreedy(k, epsilon=0.10, seed=s), {"epsilon": 0.10}),
        "thompson_sampling_beta": (lambda s: ThompsonSamplingBeta(k, alpha0=1, beta0=1, seed=s), {"prior_alpha0": 1, "prior_beta0": 1}),
    }
    for nome, (mk, params) in politicas.items():
        runs = run_experiment(mk, P, seeds=SEEDS)
        with mlflow.start_run(run_name=nome):
            mlflow.set_tags({"tipo": "politica", "familia": nome.split("_")[0], "escolhida": str(nome.startswith("thompson"))})
            mlflow.log_params({**comuns, **params})
            mlflow.log_metrics(_metricas_politica(runs, conv_banco, oraculo))
            if nome.startswith("thompson"):
                mlflow.log_artifact(str(POLICY_PATH))
                mlflow.log_artifact(str(MODELS_DIR / "metrics.json"))
        print(f"registrado: {nome}")
    print(f"\nexperimento '{EXPERIMENT}' em ./mlruns — veja com: uv run mlflow ui")


if __name__ == "__main__":
    main()
