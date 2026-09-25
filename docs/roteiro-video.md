# Roteiro do vídeo pitch (até 5 minutos)

Peso da avaliação: 30% negócio (blocos 1 e 5), 70% técnico (blocos 2, 3 e 4). Sem slides: só terminal, notebook e navegador. Ensaie uma vez com cronômetro; o texto abaixo fala em ritmo normal em ~4m30.

## Antes de gravar (checklist)

- [ ] `uv sync` feito; `models/` gerado (notebooks 02 e 03 executados).
- [ ] Terminal 1: `uv run uvicorn datathon.api:app` já rodando.
- [ ] Terminal 2: `uv run mlflow ui` já rodando.
- [ ] Navegador com 3 abas: `localhost:8000/` (página de demo), `localhost:5000`, README no GitHub.
- [ ] Jupyter aberto no notebook `02_baseline_bandit`, rolado até o gráfico "Regra fixa: depende de acertar o braço".
- [ ] Fonte do terminal aumentada; notificações desligadas.

---

## Bloco 1 · O problema (0:00 – 0:45) — tela: README, seção "Visão do problema"

> Um banco digital liga para clientes para oferecer um depósito a prazo. Hoje a decisão de **como** abordar, por qual canal e em que dia, segue regras fixas. Regras fixas dependem de alguém ter acertado de primeira e não reagem quando o contexto muda.
>
> Nosso projeto troca a regra fixa por um sistema que **aprende com cada resposta**: um multi-armed bandit. Cada braço é uma combinação de canal e dia da semana, dez no total. A recompensa é o cliente dizer sim.

## Bloco 2 · Dados e o cuidado que a banca procura (0:45 – 1:30) — tela: README, seção "Base de dados"

> Usamos a base Telemarketing JYB do Kaggle, derivada da UCI Bank Marketing: 28 mil ligações reais de um banco português, com 11,5% de conversão.
>
> Dois cuidados. Primeiro, a coluna de duração da ligação não entra: ela só é conhecida depois da chamada e entregaria a resposta ao modelo. Segundo, o mês parece o melhor sinal da base, março converte 49%, mas descobrimos que ele só marca o período de juros baixos. Usar mês como decisão ensinaria uma ilusão. Então mês e economia entram como contexto, nunca como ação.

## Bloco 3 · Baseline vs. bandit (1:30 – 3:00) — tela: notebook 02, gráfico das regras fixas

> Como avaliar sem poder ligar de novo? A base só registra o que o banco fez. Treinamos um modelo que estima a chance de conversão de cada cliente em cada braço, validamos em hold-out com AUC 0,79 e calibração, e ele vira o simulador. Uma prova de que ele é fiel: a política real do banco dá 11,45% simulado contra 11,46% observado.
>
> [apontar as barras] Estas são as dez regras fixas possíveis. Rendem de 7,6% a 12,7%, dependendo do braço que alguém escolheu de antemão. [apontar a linha laranja] Este é o Thompson Sampling: 12,1%, sem precisar adivinhar, superando oito das dez regras e a política atual do banco em 0,6 ponto. São cerca de 60 vendas a mais a cada 10 mil ligações.
>
> Escolhemos Thompson Sampling e não epsilon-greedy porque ele não tem parâmetro de exploração: começa com uma crença uniforme por braço, Beta(1,1), e a exploração diminui sozinha conforme as crenças se estreitam. [rolar até o gráfico de participação dos braços] Aqui se vê: no início espalha, depois concentra nos braços de celular.

## Bloco 4 · Demonstração ao vivo (3:00 – 4:15) — tela: navegador, `localhost:8000/`

> A política final são dez pares alfa e beta, salvos em JSON, e a API carrega isso na subida. [mostrar o formulário] O cliente chega com o contexto: perfil, histórico, indicadores econômicos. Canal e dia **não** são entrada, são a resposta.
>
> [clicar no preset "A · Já aceitou antes", depois Recomendar] Resposta: celular na quarta, propensão de 87%, prioridade alta, porque este cliente já aceitou a campanha anterior. [clicar Recomendar mais 4 ou 5 vezes] Às vezes vem celular na terça com a etiqueta "exploração", e o histórico embaixo mostra a distribuição: é o bandit continuando a aprender.
>
> [preset "D · Insistência em juros altos", Recomendar] Mesmo braço, propensão de 5%, prioridade baixa: o banco sabe que essa ligação vale pouco. [preset "E", Recomendar] E aqui o modelo avisa que, para este idoso, telefone fixo renderia mais: o limite da política sem contexto, que a gente assume.

## Bloco 5 · MLflow, nuvem e limites (4:15 – 5:00) — tela: `localhost:5000`, depois README "Arquitetura"

> [MLflow] Cada política é uma execução com seus parâmetros, o prior documentado e as métricas; o Thompson Sampling leva o arquivo da política como artefato.
>
> [README] Em produção na AWS: a API no Fargate atrás do API Gateway, o estado do bandit no DynamoDB, e o resultado de cada ligação volta por SQS e Lambda para atualizar as crenças. Esse retorno é o que faz o sistema aprender de verdade.
>
> Limitação honesta: nossa política não personaliza o braço por cliente; o oráculo mostra até 1,3 ponto adicional para uma versão contextual, que fica como próximo passo. Obrigado.

---

## Cliente A para colar na demo

```json
{
  "age": 36, "job": "management", "marital": "divorced", "education": "university.degree",
  "default": "no", "housing": "yes", "loan": "no", "month": "jun",
  "campaign": 1, "pdays": 3, "previous": 4, "poutcome": "success",
  "emp_var_rate": -1.7, "cons_price_idx": 94.055, "cons_conf_idx": -39.8,
  "euribor3m": 0.72, "nr_employed": 4991.6}
```

## Se sobrar tempo, perguntas prováveis da banca

- **Por que não regressão logística no simulador?** Mesmo AUC, mas aponta o mesmo braço para 100% dos clientes; sem interações não há o que personalizar.
- **Por que o bandit não vence a melhor regra fixa?** Os dois melhores braços diferem 0,17 pp; separá-los exigiria 311 mil ligações por braço. O bandit explora corretamente sob incerteza real.
- **Por que a política no DynamoDB e o modelo no S3?** A política muda a cada ligação (dez números, leitura e escrita frequentes); o modelo muda por mês (arquivo, só leitura).
