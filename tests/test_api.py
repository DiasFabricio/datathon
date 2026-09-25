import pytest
from fastapi.testclient import TestClient

from datathon.api import app
from datathon.data import ARMS
from datathon.recommender import MODELS_DIR, POLICY_PATH, REWARD_MODEL_PATH
from tests.test_recommender import CLIENTE

precisa_artefatos = pytest.mark.skipif(
    not (POLICY_PATH.exists() and REWARD_MODEL_PATH.exists()),
    reason="artefatos em models/ não gerados (rode os notebooks 02 e 03)",
)


@precisa_artefatos
def test_health_e_policy():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        pol = c.get("/policy").json()
        assert set(pol) == set(ARMS) and all(0 < v["mean"] < 1 for v in pol.values())


@precisa_artefatos
def test_recommend_deterministico():
    with TestClient(app) as c:
        r = c.post("/recommend?sample=false", json=CLIENTE)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["arm"] in ARMS and body["explore"] is False
        assert body["arm"] == f"{body['contact']}_{body['day_of_week']}"
        assert 0 <= body["propensity"] <= 1


@precisa_artefatos
def test_recommend_valida_entrada():
    with TestClient(app) as c:
        ruim = {**CLIENTE, "job": "astronauta"}
        assert c.post("/recommend", json=ruim).status_code == 422
        faltando = {k: v for k, v in CLIENTE.items() if k != "euribor3m"}
        assert c.post("/recommend", json=faltando).status_code == 422


@precisa_artefatos
def test_pagina_de_demonstracao():
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "Recomendar" in r.text
