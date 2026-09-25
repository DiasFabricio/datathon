"""API de recomendação (Etapa 5).

    uv run uvicorn datathon.api:app --reload
    → http://127.0.0.1:8000/docs  (documentação interativa gerada pelo FastAPI)

Recebe os dados de um cliente e devolve por qual canal e em que dia ligar
(decisão do Thompson Sampling) e a propensão de conversão (modelo de recompensa).
Usa exatamente a mesma função `recommend` do notebook 03 (Golden Set).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from datathon import __version__
from datathon.data import ARMS
from datathon.recommender import MODELS_DIR, POLICY_PATH, REWARD_MODEL_PATH, load_artifacts, recommend

# Valores permitidos: os mesmos observados na base (ver notebook 01_eda)
Job = Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
              "self-employed", "services", "student", "technician", "unemployed", "unknown"]
Marital = Literal["divorced", "married", "single", "unknown"]
Education = Literal["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                    "professional.course", "university.degree", "unknown"]
YesNoUnknown = Literal["no", "yes", "unknown"]
Month = Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
Poutcome = Literal["failure", "nonexistent", "success"]


class Cliente(BaseModel):
    """Contexto do cliente. Canal e dia NÃO entram: são a decisão da política."""

    age: int = Field(ge=17, le=100, examples=[36])
    job: Job = "admin."
    marital: Marital = "married"
    education: Education = "university.degree"
    default: YesNoUnknown = "no"
    housing: YesNoUnknown = "no"
    loan: YesNoUnknown = "no"
    month: Month = "may"
    campaign: int = Field(ge=1, description="contatos nesta campanha, incluindo o atual", examples=[1])
    pdays: int = Field(ge=0, le=999, description="dias desde o último contato; 999 = nunca contatado", examples=[999])
    previous: int = Field(ge=0, description="contatos em campanhas anteriores", examples=[0])
    poutcome: Poutcome = "nonexistent"
    emp_var_rate: float = Field(examples=[1.1])
    cons_price_idx: float = Field(examples=[93.994])
    cons_conf_idx: float = Field(examples=[-36.4])
    euribor3m: float = Field(examples=[4.857])
    nr_employed: float = Field(examples=[5191.0])


class Recomendacao(BaseModel):
    arm: str = Field(description="braço escolhido: canal_dia")
    contact: str
    day_of_week: str
    propensity: float = Field(description="P(conversão | cliente, braço recomendado)")
    policy_mean: float = Field(description="taxa média que a política acredita para o braço")
    explore: bool = Field(description="True se o braço sorteado não é o de maior média (exploração)")
    best_arm_for_client: str
    best_arm_propensity: float


STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # carrega os artefatos uma vez, na subida do servidor
    STATE["policy"], STATE["reward_model"] = load_artifacts(MODELS_DIR)
    yield
    STATE.clear()


app = FastAPI(
    title="Datathon MLET · Recomendação de abordagem (Multi-Armed Bandit)",
    description="Dado o contexto de um cliente, decide por qual canal e em que dia da semana ligar.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "arms": ARMS, "policy_file": POLICY_PATH.name, "reward_model_file": REWARD_MODEL_PATH.name}


@app.get("/policy")
def policy_state():
    """Crenças atuais do Thompson Sampling (transparência)."""
    p = STATE["policy"]
    return {a: {"alpha": float(p.alpha[i]), "beta": float(p.beta[i]), "mean": float(p.means[i])} for i, a in enumerate(ARMS)}


@app.post("/recommend", response_model=Recomendacao)
def recomendar(cliente: Cliente, sample: bool = Query(True, description="True = Thompson Sampling com sorteio; False = braço de maior média")):
    try:
        return recommend(cliente.model_dump(), STATE["policy"], STATE["reward_model"], sample=sample)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
