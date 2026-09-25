# Datathon MLET — Experimentação adaptativa com Multi-Armed Bandit

> Pós Tech FIAP · Machine Learning Engineering · Fase 05

## Visão do problema

_(a preencher na Etapa 0)_

## Base de dados

**Kaggle:** [Telemarketing JYB Dataset (Víctor Aguado)](https://www.kaggle.com/datasets/aguado/telemarketing-jyb-dataset) — versão 1, publicada em 2022-12-02.

Derivada da base **UCI Bank Marketing** (Moro, Cortez e Rita, 2014): campanhas de telemarketing de um banco português para venda de depósito a prazo. O autor já removeu a coluna `duration` (vazamento temporal) e dividiu a base em `train.csv` (28.645 linhas, com alvo `y`) e `test.csv` (12.543 linhas, sem alvo).

| Item | Valor |
|---|---|
| Arquivos | `data/raw/train.csv`, `data/raw/test.csv` (separador `;`) |
| Features | 19 (7 do cliente, 3 do contato, 4 da campanha, 5 macroeconômicas) |
| Alvo | `y` (yes/no) — cliente assinou o depósito a prazo |
| Taxa de conversão | 11,46% (desbalanceado) |
| Licença | "Other" no Kaggle; a base original UCI é CC BY 4.0 |

Cuidados documentados na EDA: coluna `Unnamed: 0` é o índice da base UCI e é descartada; `pdays = 999` significa "nunca contatado"; a categoria `unknown` funciona como valor ausente; `test.csv` não tem alvo, então serve apenas como conjunto de clientes inéditos para a demonstração da API.

## Como executar localmente

```bash
# 1. Instalar o uv (uma vez): https://docs.astral.sh/uv/
# 2. Criar ambiente e instalar dependências
uv sync

# 3. Baixar a base do Kaggle (download público, não exige conta)
curl -L -o data/raw/dataset.zip https://www.kaggle.com/api/v1/datasets/download/aguado/telemarketing-jyb-dataset \
  && unzip -o data/raw/dataset.zip -d data/raw && rm data/raw/dataset.zip

# 4. Abrir os notebooks
uv run jupyter lab

# 5. Rodar os testes
uv run pytest

# 6. Subir a API
uv run uvicorn datathon.api:app --reload
```

## Estrutura do repositório

```
├── data/            # raw/ (download Kaggle) e processed/ — ignorados pelo git
├── docs/            # enunciado do desafio
├── notebooks/       # EDA, baseline, bandit, avaliação
├── src/datathon/    # pacote Python: dados, políticas (bandits), API
├── tests/           # testes automatizados
└── pyproject.toml   # dependências (gerenciadas com uv)
```

## Baseline vs. política adaptativa

Notebook: `notebooks/02_baseline_bandit.ipynb`.

**Formulação.** Braços = 10 combinações de canal (`contact`) × dia da semana (`day_of_week`), as duas variáveis que o banco controla antes de ligar. Recompensa = cliente assinou o depósito (`y == yes`). Contexto (perfil, histórico, mês, macroeconomia) entra no simulador; `month` não é braço porque está confundido com o período econômico (ver EDA).

**Avaliação offline.** A base só registra o braço que o banco usou. Um modelo de recompensa (gradient boosting, AUC 0,79 em hold-out, calibrado) estima P(conversão | contexto, braço) e serve de simulador: a política escolhe, a recompensa é sorteada. Todas as políticas rodam sobre os mesmos 28.645 clientes, com 10 sementes e sorteios comuns.

| Política | Conversão simulada | vs. baseline |
|---|---|---|
| Política atual do banco (**baseline**: regras fixas do status quo) | 11,45% | — |
| Epsilon-Greedy ε = 0,05 | 12,28% ± 0,42 | +0,8 pp |
| **Thompson Sampling Beta(1,1)** (política escolhida) | 12,06% ± 0,17 | +0,6 pp |
| Regra fixa "melhor braço histórico" (referência forte) | 12,58% ± 0,17 | exige acertar o braço de primeira |
| Oráculo (teto teórico) | 13,85% | — |

**Por que Thompson Sampling.** Prior Beta(1,1) por braço (uniforme, documentado). Sem parâmetro de exploração: a exploração diminui sozinha conforme as crenças se estreitam. Supera 8 das 10 regras fixas possíveis (que rendem de 7,6% a 12,7% conforme o braço escolhido), ou seja, entrega o resultado sem depender de um acerto inicial. O Epsilon-Greedy rendeu um pouco mais, mas com variância 2,5x maior e um ε escolhido à mão.

**Limitações.** (1) Recompensas de braços não observados vêm do simulador. (2) Os braços têm efeito médio muito próximo (0,17 pp entre os dois melhores); separá-los exigiria ≈311 mil ligações por braço, então o bandit sem contexto não supera a melhor regra fixa possível. Uma variação contextual está fora do escopo.

## Avaliação e Golden Set

Notebook: `notebooks/03_avaliacao_golden_set.ipynb`. A política final (crenças Beta após 28.645 interações simuladas) fica em `models/policy_thompson.json`; as métricas em `models/metrics.json`.

| Métrica | Valor |
|---|---|
| Modelo de recompensa: AUC / Brier (hold-out 30%) | 0,789 / 0,079 (média fixa: 0,102) |
| Conversão Thompson Sampling (10 sementes) | 12,06% ± 0,17 |
| Conversão política atual do banco (baseline) | 11,45% |
| Lift absoluto / relativo | +0,61 pp / +5,3% |
| Conversões adicionais por 10 mil ligações | ≈61 |
| Fração do ganho possível capturada (banco → oráculo) | 25% |

**Como o contexto entra.** Na avaliação (simulador) e na priorização (propensão devolvida pela API). Não na escolha do braço: Thompson Sampling por segmento (`poutcome`, `contato_previo`) foi testado e ficou abaixo do global (11,9% vs 12,1%), porque o melhor braço é o mesmo em todos os segmentos e dividir os dados só encarece a exploração.

**Golden Set** (5 clientes inéditos de `test.csv`; recomendação = braço de maior média posterior; propensão = P(conversão | cliente, braço)):

| Cliente | Perfil | Recomendação | Propensão | Faz sentido? |
|---|---|---|---|---|
| A | Gerente, 36 anos, aceitou a campanha anterior, juros baixos | celular / quarta | 87% | Sim: perfil mais valioso da base (64% de conversão histórica para `success`). Prioridade máxima |
| B | Estudante, 23 anos, sem dívidas, nunca contatado, março/juros baixos | celular / quarta | 62% | Sim: jovem sem dívidas no período em que o produto era atraente |
| C | Aposentado, 74 anos, recusou a campanha anterior | celular / quarta | 68% | Sim, com ressalva: já contatados convertem mais que nunca contatados; o modelo indica terça um pouco melhor (74%) |
| D | Operário, 59 anos, 6 contatos nesta campanha, juros altos | celular / quarta | 5% | Sim: insistir não ajuda (EDA) e o período é ruim. Prioridade baixa |
| E | Aposentado, 77 anos, aceitou a campanha anterior | celular / quarta | 80% | Sim, com alerta: o modelo estima telefone fixo melhor para este idoso (84%); limite da política global |

A política recomenda o mesmo braço a todos porque ele é o melhor em média; a propensão (5% a 87%) é o que diferencia os clientes e orienta a priorização. Em operação, o Thompson Sampling sorteia: ≈80% das chamadas vão aos dois braços empatados na liderança e ≈20% exploram os demais.

## API de recomendação (Etapa 5)

Código: `src/datathon/api.py` (FastAPI). Usa a mesma função `datathon.recommender.recommend` do Golden Set, garantindo que o que foi avaliado é o que está sendo servido.

```bash
uv run uvicorn datathon.api:app --reload
# documentação interativa: http://127.0.0.1:8000/docs
```

| Endpoint | Função |
|---|---|
| `GET /health` | status e lista de braços |
| `GET /policy` | crenças atuais (α, β, média) do Thompson Sampling por braço |
| `POST /recommend` | recebe o contexto do cliente e devolve canal, dia, propensão e se a decisão foi exploratória. `?sample=false` devolve o braço de maior média (determinístico) |

Exemplo:

```bash
curl -s -X POST "http://127.0.0.1:8000/recommend" -H "Content-Type: application/json" -d '{
  "age": 36, "job": "management", "marital": "divorced", "education": "university.degree",
  "default": "no", "housing": "yes", "loan": "no", "month": "jun",
  "campaign": 1, "pdays": 3, "previous": 4, "poutcome": "success",
  "emp_var_rate": -2.9, "cons_price_idx": 92.963, "cons_conf_idx": -40.8,
  "euribor3m": 0.72, "nr_employed": 5076.2}'
```

```json
{"arm": "cellular_wed", "contact": "cellular", "day_of_week": "wed",
 "propensity": 0.67, "policy_mean": 0.126, "explore": false,
 "best_arm_for_client": "cellular_fri", "best_arm_propensity": 0.67}
```

Entradas fora do domínio (ex.: `job = "astronauta"`) são rejeitadas com HTTP 422 pela validação do Pydantic. Os artefatos em `models/` são gerados pelos notebooks 02 e 03; sem eles a API não sobe.

## Arquitetura-alvo em nuvem (AWS)

A operação tem dois caminhos. No **caminho de decisão**, síncrono, o canal (app, central de atendimento) chama a API FastAPI, empacotada em container e executada no **ECS Fargate** atrás do **API Gateway**, e recebe em milissegundos o canal e o dia recomendados mais a propensão do cliente. O container baixa o modelo de recompensa do **S3** na subida, mas não guarda a política em arquivo: as dez contagens de sucessos e falhas do Thompson Sampling ficam em uma tabela no **DynamoDB**, para que todas as réplicas da API leiam o mesmo estado e nenhuma atualização se perca.

No **caminho de aprendizado**, assíncrono, o resultado de cada ligação (converteu ou não) é publicado como evento em uma fila **SQS**; uma **Lambda** consome a fila e incrementa alfa ou beta do braço correspondente no DynamoDB, que é exatamente o método `update` deste repositório rodando como função. É esse retorno que faz o sistema aprender com as respostas observadas em vez de congelar em regras fixas. O **CloudWatch** acompanha conversão por braço, latência da API e a fração de decisões exploratórias, com alerta se um braço deixar de ser escolhido ou se a exploração cair a zero. Os eventos ficam no S3 para o retreino periódico do modelo de recompensa, versionado no MLflow. Em governança: nenhum identificador direto do cliente trafega nos eventos, a lista de braços permitidos só muda com aprovação humana, e a retenção dos eventos é definida por finalidade.

| Peça do repositório | Serviço AWS |
|---|---|
| `datathon.api` (FastAPI) | ECS Fargate + API Gateway |
| `models/reward_model.joblib`, dados brutos e eventos | S3 |
| `models/policy_thompson.json` (α, β por braço) | DynamoDB |
| `ThompsonSamplingBeta.update` | SQS + Lambda |
| Métricas e alertas | CloudWatch |
| Parâmetros e métricas de treino | MLflow (Etapa 7) |

## MLOps — tracking com MLflow (Etapa 7)

Script: `src/datathon/tracking.py`. Registra no experimento `datathon-bandit-telemarketing`, em `./mlruns` e `mlflow.db` (locais, ignorados pelo git), uma execução por política e uma para o modelo de recompensa:

```bash
uv run python -m datathon.tracking   # registra (reexecuta as simulações, ~30 s)
uv run mlflow ui                      # http://127.0.0.1:5000
```

| Execução | Parâmetros registrados | Métricas registradas |
|---|---|---|
| `reward_model_hgb` | `max_iter`, `learning_rate`, `max_leaf_nodes`, `random_state` | AUC e Brier em hold-out |
| `baseline_politica_banco` | dataset, nº de clientes e braços | conversão, oráculo |
| `epsilon_greedy_0.05` / `_0.10` | `epsilon`, nº de sementes, método de avaliação | conversão média e desvio, lift em pp, conversões extra por 10 mil, regret final, fração do ganho capturada |
| `thompson_sampling_beta` (escolhida) | `prior_alpha0 = 1`, `prior_beta0 = 1` | as mesmas, mais os artefatos `policy_thompson.json` e `metrics.json` |

Cada execução leva a tag `dataset` com a referência do Kaggle, para rastrear de qual base saíram os números.
