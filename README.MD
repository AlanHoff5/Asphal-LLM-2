# Asphal-LLM

**Construção de um LLM From Scratch**

Projeto Integrador da disciplina de Inteligência Artificial e Sistemas Inteligentes — Engenharia de Computação, Unoesc Joaçaba.

O objetivo é implementar do zero os principais componentes de um modelo de linguagem baseado em Transformer, seguindo o livro *Build a Large Language Model (From Scratch)*, do Sebastian Raschka.

A ideia não é competir com modelos comerciais, e sim entender na prática como cada peça funciona: tokenização, embeddings, mecanismos de atenção, blocos transformer, treinamento e geração de texto.

**Integrantes:** Alan Hoffmann Dos Santos e Jackson
**Professor:** Kleyton Hoffmann
**Fase:** 2026/2

## Estrutura do projeto

```
Asphal-LLM-2/
├── data/            # datasets usados nos experimentos (ex: the-verdict.txt)
├── docs/            # análise dos resultados de cada Sprint
│   └── sprintN_analise.md
├── glossario/       # glossário técnico, um arquivo por capítulo do livro
│   └── chapterN.md
├── notebooks/       # notebook único e cumulativo do projeto
│   └── InteligenciaArtificial_SistemasInteligentes.ipynb
├── src/             # código dos componentes, organizado por módulo
│   └── tokenizer.py, attention.py, gpt_model.py, ...
├── tests/           # testes automatizados (pytest), um arquivo por componente/sprint
│   └── test_pipeline.py
├── requirements.txt
└── README.MD
```

**Convenção para as próximas Sprints** (evita ficar perdido conforme o projeto cresce):

- **Glossário**: sempre `glossario/chapterN.md`, um arquivo por capítulo do livro, atualizado de forma cumulativa.
- **Análise**: sempre `docs/sprintN_analise.md`, respondendo às perguntas técnicas do roteiro daquela Sprint.
- **Código**: cada componente novo (atenção, bloco transformer, modelo GPT, loop de treinamento, fine-tuning) vira um módulo próprio dentro de `src/`, importável (`from src.<modulo> import ...`), nunca só dentro do notebook.
- **Testes**: cada módulo novo em `src/` ganha um `tests/test_<modulo>.py` correspondente, seguindo o mesmo padrão de `test_pipeline.py` (um teste por etapa do fluxo daquele componente).
- **Notebook**: continua sendo um único notebook cumulativo, com uma seção numerada por capítulo/sprint (ex: `2.9 Experimentação`, `3.x ...`), igual ao padrão já usado no Capítulo 2.
- **Dados**: qualquer arquivo de texto/dataset baixado para os experimentos vai para `data/`, nunca solto na raiz do projeto ou dentro de `notebooks/`.

## Como rodar

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

Rodar os testes:

```bash
python -m pytest tests/ -v
```

Abrir o notebook:

```bash
jupyter lab notebooks/InteligenciaArtificial_SistemasInteligentes.ipynb
```

## Progresso das Sprints

### Sprint 0 — Preparação do Ambiente

| Entregável | Status |
|---|---|
| Ambiente computacional configurado (`.venv`, `requirements.txt`) | ✅ |
| `README.MD` com descrição do projeto | ✅ |
| Glossário do Capítulo 1 (`glossario/chapter1.md`) | ✅ |
| Repositório do projeto (GitHub/GitLab) | ⬜ pendente — a cargo do grupo |

### Sprint 2 — Trabalhando com Dados Textuais

| Entregável | Status |
|---|---|
| Glossário do Capítulo 2 (`glossario/chapter2.md`) | ✅ |
| Código de tokenização (`src/tokenizer.py`) | ✅ |
| Vocabulário e Token IDs | ✅ |
| Preparação das sequências de treinamento | ✅ |
| Embeddings | ✅ |
| Positional Embeddings | ✅ |
| Estrutura de carregamento dos dados (DataLoader) | ✅ |
| Experimentos (`notebooks/...ipynb`, seção 2.9) | ✅ |
| Análise dos resultados (`docs/sprint2_analise.md`) | ✅ |
| Testes automatizados (`tests/test_pipeline.py`) | ✅ |
| Código organizado no repositório | ✅ |
