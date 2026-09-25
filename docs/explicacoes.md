# Explicações do projeto — Datathon MLET

_Documento vivo. Atualizado a cada etapa. Última revisão: 25/09/2026 (Etapas 1 a 7 concluídas; roteiro do vídeo pronto)._

## O que é o projeto, em uma frase

Um sistema que aprende, ligação a ligação, **por qual canal e em que dia da semana abordar cada cliente** para vender um depósito a prazo, em vez de seguir uma regra fixa.

## Por que isso é um "multi-armed bandit"

Imagine um cassino com 10 máquinas (braços). Se puxar sempre a mesma, nunca descobre se outra paga mais; se testar todas o tempo todo, desperdiça fichas. O bandit equilibra: **explora** um pouco para aprender e **explota** o que já sabe que funciona. Aqui, cada máquina é uma combinação de canal e dia, cada ficha é uma ligação, e o prêmio é o cliente dizer sim.

## A base de dados

Telemarketing JYB Dataset (Kaggle, Víctor Aguado), derivada da UCI Bank Marketing: campanhas telefônicas de um banco português entre 2008 e 2010.

| Item | Valor |
|---|---|
| Linhas com resultado conhecido | 28.645 |
| Linhas sem resultado (só para demonstração) | 12.543 |
| Variáveis | 19 (perfil, contato, histórico, economia) |
| Taxa de conversão | 11,5% |

## O que encontramos na análise exploratória

1. **O canal importa muito.** Celular converte 15%, telefone fixo 5%.
2. **O dia da semana importa pouco.** Quinta 12,2%, segunda 10,4%: sinal fraco que regras fixas ignoram.
3. **O mês engana.** Março converte 49% e maio 7%, mas os meses "bons" tiveram poucas ligações e caíram na fase de juros baixos. O mês marca a época, não causa a venda.
4. **Histórico é o maior sinal.** Quem já aceitou uma campanha anterior converte 64%. Quem nunca foi contatado, 9%.
5. **A base é limpa, mas tem pegadinhas.** 21% dos clientes têm inadimplência "desconhecida"; o código 999 em dias desde o último contato significa "nunca contatado"; há 890 linhas idênticas sem identificador de cliente.
6. **A coluna de duração da ligação não existe**, e isso é bom: ela só é conhecida depois da ligação e entregaria a resposta ao modelo.

## Decisões tomadas até agora

| Decisão | Escolha | Motivo |
|---|---|---|
| Ações do bandit (braços) | canal x dia da semana, 10 combinações | São as variáveis que o banco controla antes de ligar; cada braço tem 2 mil ou mais ligações |
| Contexto | perfil do cliente, histórico, mês e indicadores econômicos | O bandit observa, mas não escolhe |
| Recompensa | cliente disse sim (1) ou não (0) | Alvo direto da base |
| Baseline para comparação | a política atual do banco (o que ele de fato fez), 11,45% | É o status quo com regras fixas que o enunciado critica; "melhor braço histórico" fica como referência mais forte |
| Como avaliar sem poder ligar de novo | simulador que estima a chance de conversão de cada cliente em cada braço | A base só registra o que o banco fez; o simulador reproduz a taxa real |
| Algoritmo adaptativo | Thompson Sampling | Sem parâmetro para ajustar, exploração diminui sozinha, padrão de mercado |
| O que a API devolve | canal + dia (decisão do bandit) e a propensão do cliente (modelo de recompensa) | O bandit decide como ligar; a propensão diz para quem ligar primeiro |
| Versão contextual | fora do escopo | O enunciado pede o básico; testamos e a versão simples piora por falta de volume |
| Duplicatas e valores "desconhecido" | mantidos | Sem identificador, cada linha é um contato real; não saber é informação |
| Limpeza dos dados | função única no pacote, com testes automatizados | Notebook e API usam a mesma transformação; ninguém copia célula |
| Formato de entrega | notebook para análise, pacote Python reutilizável, API FastAPI para demonstração | Peso técnico é 70% da nota |

## O que já está pronto

- EDA, base tratada e comparação baseline vs. bandit (notebooks `01` e `02`). O Thompson Sampling converte 12,1% contra 11,45% da política atual do banco: cerca de 170 vendas a mais a cada 28 mil ligações, aprendendo só com as respostas.
- Achado honesto: a melhor regra fixa possível ("sempre celular na terça") chega a 12,6%. Mas as dez regras fixas possíveis rendem de 7,6% a 12,7% conforme o braço escolhido; o bandit entrega 12,1% sem precisar adivinhar e supera oito delas.

- Política final e 5 clientes de teste avaliados (notebook `03`). A decisão fez sentido nos cinco; em dois, outro canal seria um pouco melhor, o limite de uma política sem contexto.

- API no ar: recebe os dados de um cliente e devolve canal, dia e propensão. Rejeita entradas inválidas. Mesma função usada nos 5 casos de teste.

- Arquitetura em nuvem descrita no README (AWS): API no Fargate, estado do bandit no DynamoDB, resultado de cada ligação volta por SQS e Lambda para atualizar as crenças. Esse retorno é o que faz o sistema aprender em produção.

- Experimentos registrados no MLflow: uma execução por política, com parâmetros e métricas, e o arquivo da política anexado à execução escolhida.

## O que ainda falta

- Etapa 8: gravar o vídeo de 5 minutos (roteiro em `docs/roteiro-video.md`).
- Publicar o repositório no GitHub.

## Como rodar o que existe hoje

```bash
uv sync
uv run jupyter lab            # notebooks 01, 02 e 03
uv run uvicorn datathon.api:app --reload   # API em http://127.0.0.1:8000/docs
```

A base é baixada automaticamente do Kaggle sem precisar de conta (instruções no README).
