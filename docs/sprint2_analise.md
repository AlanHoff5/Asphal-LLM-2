# Sprint 2 — Análise dos Resultados

Análise técnica referente à Seção 5 do roteiro da Sprint 2 ("Trabalhando com Dados Textuais"). Os números citados vêm diretamente dos experimentos executados na seção **2.9** do notebook [`InteligenciaArtificial_SistemasInteligentes.ipynb`](../notebooks/InteligenciaArtificial_SistemasInteligentes.ipynb), sobre o texto `the-verdict.txt` (20.479 caracteres, 4.690 tokens por palavra / 5.145 tokens BPE).

## 1. Por que um LLM não pode trabalhar diretamente com o texto bruto?

Uma rede neural opera sobre operações matriciais (multiplicações, somas, gradientes), que exigem valores numéricos contínuos como entrada — não existe uma operação de "multiplicar por uma palavra". O texto bruto (Raw Text) é uma sequência de caracteres sem estrutura numérica nenhuma. Por isso é necessário todo o pipeline desta Sprint: **Tokenização → Token IDs → Embeddings**, que transforma progressivamente texto em algo que a rede consegue processar. No Experimento 1, o mesmo texto (`the-verdict.txt`, 20.479 caracteres) só se torna utilizável depois de virar 5.145 Token IDs via BPE — o modelo nunca "lê" as 20.479 letras, ele processa aquela sequência de 5.145 inteiros.

## 2. Qual é a função do vocabulário?

O vocabulário é o dicionário fechado que define a relação bijetora entre cada token conhecido e um identificador numérico único (Token ID), nas duas direções (`str_to_int` e `int_to_str` no `SimpleTokenizerV2`). Ele delimita o "alfabeto" que o tokenizador consegue produzir. No Experimento 2 isso fica visível: o vocabulário por palavras construído sobre `the-verdict.txt` tem 1.132 entradas, e é justamente por causa dessa limitação que a palavra "Hello" (fora do vocabulário) gera `KeyError` no `SimpleTokenizerV1` e precisa do tratamento por `<|unk|>` no V2. O BPE contorna esse limite tendo um vocabulário fixo de 50.257 subpalavras, capaz de recombinar qualquer palavra nova sem nunca precisar de um token de desconhecido.

## 3. Qual é a diferença entre um token e um Token ID?

O **token** é a unidade textual em si — uma palavra, subpalavra ou sinal de pontuação (ex.: `"Hello"`, `","`, `"world"`). O **Token ID** é o inteiro que o vocabulário associa a esse token, usado internamente pelo modelo. Um existe apenas como referência humana; o outro é o que efetivamente circula no pipeline computacional. Isso fica claro no Experimento 2: a frase `"In the sunlit terraces of the palace, Mrs. Gisburn said with pardonable pride."` vira os tokens (palavras) mapeados para os IDs `[55, 988, 956, 984, 722, 988, 1131, 5, 67, 7, 38, 851, 1108, 754, 793, 7]` — cada posição da lista é um Token ID, não o texto do token.

## 4. Por que os Token IDs não são utilizados diretamente como representação semântica?

Porque um Token ID é apenas um rótulo arbitrário de posição no vocabulário — o valor numérico não carrega nenhuma noção de significado ou semelhança. No vocabulário por palavras construído no notebook, `"Gisburn"` tem ID 38 e `"Gisburns"` tem ID 39 simplesmente por ordem alfabética; não há relação matemática entre 38 e 39 que reflita que as palavras são relacionadas. Se o modelo usasse o ID bruto como entrada numérica, ele interpretaria como se o ID 39 fosse "um pouco maior" que o ID 38 — uma relação de ordem que não existe semanticamente. É exatamente para resolver isso que existe a camada de Embedding (pergunta 5).

## 5. Qual é a função dos embeddings?

A camada de Embedding (`torch.nn.Embedding`) substitui cada Token ID discreto por um vetor denso e treinável, onde cada dimensão pode capturar algum aspecto latente de significado — e essa representação pode ser ajustada via backpropagation durante o treinamento, ao contrário do ID fixo. No Experimento 6, o mesmo lote de tokens gera vetores de dimensões diferentes conforme o `output_dim` escolhido: com `output_dim=256` (o valor usado no restante do capítulo), cada token de um lote `(8, 4)` passa a ser representado por um tensor `(8, 4, 256)` — 256 números contínuos por token, em vez de 1 inteiro.

## 6. Por que é necessário representar a posição dos tokens?

O mecanismo de Attention (Sprint 3) trata a sequência de forma essencialmente não-ordenada — ele calcula relações entre todos os pares de tokens, sem noção intrínseca de "antes" ou "depois". Sem uma informação adicional de posição, trocar a ordem de duas palavras não mudaria em nada a representação do modelo, o que destruiria a estrutura da linguagem (compare "o cão mordeu o homem" com "o homem mordeu o cão" — mesmos tokens, ordem diferente, significado oposto). Por isso soma-se ao Token Embedding um **Positional Embedding**, também uma tabela treinável, mas indexada pela posição (0, 1, 2, ...) e não pelo conteúdo do token. No pipeline do notebook, `pos_embeddings` tem shape `(4, 256)` para um `context_length=4`, e é somado ao `token_embeddings` `(8, 4, 256)` — via broadcasting — para produzir o `input_embeddings` final, que agora carrega tanto "o que é o token" quanto "onde ele está".

## 7. Qual é a relação entre tamanho do contexto e quantidade de amostras de treinamento?

É uma relação inversa: quanto maior o `context_size` (com `stride = context_size`, sem sobreposição), menos amostras cabem no mesmo texto, porque cada amostra "consome" mais tokens da sequência original. O Experimento 3 mostra isso de forma direta sobre os mesmos 5.145 tokens BPE de `the-verdict.txt`:

| context_size | amostras produzidas |
|---:|---:|
| 2 | 2.572 |
| 4 | 1.286 |
| 8 | 643 |
| 16 | 321 |
| 32 | 160 |
| 64 | 80 |
| 128 | 40 |

Dobrar o contexto aproximadamente divide o número de amostras pela metade (5.145/context_size). O Experimento 4 mostra a outra face dessa relação: fixando `context_size=32` e reduzindo o `stride` (aumentando a sobreposição entre janelas), o número de amostras cresce sem que o conteúdo textual mude — com `stride=8` (75% de sobreposição) obtém-se 640 amostras, contra apenas 160 com `stride=32` (0% de sobreposição). Ou seja, o stride é uma alavanca independente do context_size para controlar quantidade de dados às custas de repetição de conteúdo entre amostras.

## 8. Qual é o impacto da dimensão do embedding sobre as estruturas utilizadas pelo modelo?

A dimensão do embedding (`output_dim`) define o tamanho do último eixo de todos os tensores produzidos a partir daquele ponto do pipeline, e cresce linearmente o número de parâmetros da camada de embedding (`vocab_size × output_dim`). No Experimento 6, com o vocabulário BPE fixo em 50.257 tokens:

| output_dim | shape input_embeddings | parâmetros (token embedding) |
|---:|---|---:|
| 8 | (8, 4, 8) | 402.056 |
| 32 | (8, 4, 32) | 1.608.224 |
| 64 | (8, 4, 64) | 3.216.448 |
| 128 | (8, 4, 128) | 6.432.896 |
| 256 | (8, 4, 256) | 12.865.792 |
| 768 | (8, 4, 768) | 38.597.376 |

O número de parâmetros escala exatamente proporcional ao `output_dim` (dobrar a dimensão dobra os parâmetros), já que `vocab_size` é constante. Isso é diretamente relevante para o restante do projeto: aumentar `output_dim` melhora a capacidade de representação do modelo, mas tem custo de memória e computação — é o mesmo tipo de trade-off citado no Capítulo 1 do livro (a camada de Token Embedding do GPT-2 pequeno já usa `output_dim=768`).

## 9. Qual é a função do DataLoader no pipeline?

O `DataLoader` (via `GPTDatasetV1` + `create_dataloader_v1`) é a ponte entre o texto pré-processado (a lista de pares Input-Target extraídos pela Sliding Window) e o loop de treinamento: ele organiza essas amostras em lotes (`Batch`) de tamanho fixo, cuidando de embaralhamento (`shuffle`) e do descarte do último lote incompleto (`drop_last`). O Experimento 5 mostra como o `batch_size` afeta apenas a primeira dimensão dos tensores produzidos, sem alterar `context_size` nem a granularidade dos dados:

| batch_size | nº de lotes | shape inputs |
|---:|---:|---|
| 1 | 1.286 | (1, 4) |
| 4 | 321 | (4, 4) |
| 8 | 160 | (8, 4) |
| 16 | 80 | (16, 4) |
| 32 | 40 | (32, 4) |

Sem o DataLoader, seria necessário fatiar manualmente a lista de 1.286 amostras a cada passo de treinamento — ele automatiza exatamente essa etapa.

## 10. Quais informações produzidas nesta Sprint serão utilizadas pelo mecanismo de atenção da Sprint seguinte?

O `input_embeddings` — a soma de Token Embedding com Positional Embedding, com shape `(batch_size, context_size, output_dim)` — é a entrada direta da primeira camada de Attention da Sprint 3. É sobre esse tensor que o mecanismo de Self-Attention vai calcular a importância relativa entre cada par de posições da sequência. Também seguem adiante: o `context_size` (que define o tamanho da janela sobre a qual a atenção é calculada) e os próprios pares Input-Target gerados pela Sliding Window, que continuarão sendo a base da tarefa de Next-Word Prediction durante o treinamento do bloco completo do Transformer.

## Custo computacional (observação adicional — Experimento 7)

Como observação sobre custo, o Experimento 7 mediu o tempo médio de tokenização do texto completo pelos dois métodos: **3,77 ms** para o BPE (`tiktoken`, implementado em Rust) contra **6,39 ms** para o tokenizador por palavras em Python puro (`SimpleTokenizerV2`, baseado em regex) — cerca de 40% mais rápido, apesar de o BPE produzir mais tokens por trecho de texto (Experimento 1 e 2). Isso reforça por que os LLMs modernos preferem BPE: além de eliminar o problema de palavras fora do vocabulário, a implementação é mais eficiente computacionalmente.
