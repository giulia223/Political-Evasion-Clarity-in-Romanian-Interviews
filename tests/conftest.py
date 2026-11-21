import pytest

from polero.data import load_test, load_train


@pytest.fixture(scope="session")
def train():
    return load_train()


@pytest.fixture(scope="session")
def test_df():
    return load_test()


@pytest.fixture(scope="session")
def tiny_tok(tmp_path_factory, train):
    from polero.tiny import make_tiny_tokenizer

    texts = train["interview_question"].tolist()[:200] + train["interview_answer"].tolist()[:200]
    return make_tiny_tokenizer(texts, tmp_path_factory.mktemp("tok"), vocab_size=600)
