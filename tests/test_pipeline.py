"""
Testes de ponta a ponta do pipeline da Sprint 2

Texto -> Tokenizacao -> Tokens -> Token IDs -> Sequencias de treinamento ->
Embeddings -> Positional Embeddings -> Lote de dados -> Entrada do modelo

Cada estagio do fluxo tem seu proprio teste
"""

import tiktoken
import torch
import pytest

from tokenizer import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
    GPTDatasetV1,
    create_dataloader_v1,
)

SAMPLE_TEXT = (
    "I HAD always thought Jack Gisburn rather a cheap genius--though a "
    "good fellow enough--so it was no great surprise to me to hear that, "
    "in the height of his glory, he had dropped his painting, married a "
    "rich widow, and established himself in a villa on the Riviera. "
    "The moral pointed by his career was, of course, obvious. Others had "
    "been equally sure of their gifts, only to see them fade and vanish; "
    "but Gisburn had had the rare courage to acknowledge himself in time, "
    "and to spare his friends the pain of watching that too-common "
    "decline."
)


@pytest.fixture(scope="module")
def word_vocab():
    return build_vocab(SAMPLE_TEXT)


@pytest.fixture(scope="module")
def bpe_tokenizer():
    return tiktoken.get_encoding("gpt2")


# 1. Texto ---------------------------------------------------------------

def test_01_texto_bruto_e_uma_string_sem_estrutura_numerica():
    assert isinstance(SAMPLE_TEXT, str)
    assert len(SAMPLE_TEXT) > 0


# 2. Tokenizacao -----------------------------------------------------------

def test_02_tokenizacao_separa_palavras_e_pontuacao(word_vocab):
    tokenizer = SimpleTokenizerV2(word_vocab)
    ids = tokenizer.encode("Hello, world.")
    # "Hello" (desconhecido), ",", "world" (conhecido), "."
    assert len(ids) == 4


# 3. Tokens -> Token IDs ----------------------------------------------------

def test_03_token_id_e_bijecao_com_o_vocabulario(word_vocab):
    tokenizer = SimpleTokenizerV1(word_vocab)
    text = "Gisburn had established himself in a villa"
    ids = tokenizer.encode(text)

    for token_id in ids:
        token = tokenizer.int_to_str[token_id]
        assert tokenizer.str_to_int[token] == token_id

    assert tokenizer.decode(ids) == text


def test_03b_palavra_fora_do_vocabulario_vira_unk(word_vocab):
    tokenizer = SimpleTokenizerV2(word_vocab)
    ids = tokenizer.encode("supercalifragilisticexpialidocious")
    assert ids == [word_vocab["<|unk|>"]]


def test_03c_bpe_nunca_precisa_de_unk(bpe_tokenizer):
    # BPE quebra qualquer palavra em subpalavras/caracteres conhecidos,
    # entao nunca gera erro nem precisa de um token de desconhecido.
    ids = bpe_tokenizer.encode("supercalifragilisticexpialidocious")
    assert len(ids) > 0
    assert bpe_tokenizer.decode(ids) == "supercalifragilisticexpialidocious"


# 4. Sequencias de treinamento (Input-Target Pair via Sliding Window) ------

def test_04_alvo_e_a_entrada_deslocada_em_uma_posicao(bpe_tokenizer):
    max_length, stride = 4, 4
    token_ids = bpe_tokenizer.encode(SAMPLE_TEXT)
    dataset = GPTDatasetV1(SAMPLE_TEXT, bpe_tokenizer, max_length, stride)

    first_input, first_target = dataset[0]
    assert first_input.tolist() == token_ids[0:max_length]
    assert first_target.tolist() == token_ids[1:max_length + 1]


def test_05_quantidade_de_amostras_depende_de_context_size_e_stride(bpe_tokenizer):
    token_ids = bpe_tokenizer.encode(SAMPLE_TEXT)

    for max_length, stride in [(4, 4), (8, 8), (8, 4)]:
        dataset = GPTDatasetV1(SAMPLE_TEXT, bpe_tokenizer, max_length, stride)
        expected = len(range(0, len(token_ids) - max_length, stride))
        assert len(dataset) == expected


# 5. Embeddings --------------------------------------------------------------

def test_06_embedding_layer_produz_vetor_denso_por_token():
    vocab_size, output_dim = 50257, 16
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    ids = torch.tensor([15496, 11, 995])  # "Hello", ",", " world"
    embeddings = embedding_layer(ids)

    assert embeddings.shape == (len(ids), output_dim)
    # a mesma id deve sempre produzir o mesmo vetor (lookup determinístico)
    assert torch.equal(embedding_layer(ids[0:1]), embeddings[0:1])


# 6. Positional Embeddings ----------------------------------------------------

def test_07_positional_embedding_e_diferente_para_cada_posicao():
    context_size, output_dim = 4, 16
    torch.manual_seed(123)
    pos_embedding_layer = torch.nn.Embedding(context_size, output_dim)

    pos_embeddings = pos_embedding_layer(torch.arange(context_size))

    assert pos_embeddings.shape == (context_size, output_dim)
    # posicoes diferentes precisam gerar vetores diferentes, senao a
    # informacao de ordem se perde
    assert not torch.equal(pos_embeddings[0], pos_embeddings[1])


# 7. Lote de dados (DataLoader) -----------------------------------------------

def test_08_dataloader_organiza_amostras_em_lotes():
    batch_size, max_length = 4, 4
    dataloader = create_dataloader_v1(
        SAMPLE_TEXT, batch_size=batch_size, max_length=max_length,
        stride=max_length, shuffle=False, drop_last=True,
    )

    inputs, targets = next(iter(dataloader))

    assert inputs.shape == (batch_size, max_length)
    assert targets.shape == (batch_size, max_length)
    # drop_last=True: nenhum lote incompleto deve sobrar
    assert len(dataloader) * batch_size <= len(dataloader.dataset)


# 8. Entrada do modelo (Token Embedding + Positional Embedding) --------------

def test_09_entrada_do_modelo_combina_token_e_posicao():
    output_dim, context_size = 16, 4
    torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(50257, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_size, output_dim)

    # mesmo Token ID repetido em todas as posicoes: sem a informacao de
    # posicao, as quatro entradas seriam indistinguiveis para o modelo
    repeated_ids = torch.tensor([[42, 42, 42, 42]])

    token_embeddings = token_embedding_layer(repeated_ids)
    pos_embeddings = pos_embedding_layer(torch.arange(context_size))
    input_embeddings = token_embeddings + pos_embeddings

    assert input_embeddings.shape == (1, context_size, output_dim)
    # com o mesmo token em todas as posicoes, so a soma da Positional
    # Embedding diferencia as posicoes entre si
    assert not torch.equal(input_embeddings[0, 0], input_embeddings[0, 1])
