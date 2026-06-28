import json
import tempfile
import time
import os

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


SENNRICH_CORPUS = """\
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

SPECIAL_TOKEN = "<|endoftext|>"


def test_sennrich_first_merge():
    """First merge should be ('s', 't') — tied with ('e', 's') but lexicographically greater."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SENNRICH_CORPUS)
        tmp = f.name
    try:
        # 256 bytes + 1 special token + 1 merge = 258
        vocab, merges = run_train_bpe(
            input_path=tmp,
            vocab_size=258,
            special_tokens=[SPECIAL_TOKEN],
        )
        assert len(merges) == 1
        assert merges[0] == (b"s", b"t")
    finally:
        os.unlink(tmp)


def test_sennrich_six_merges():
    """After 6 merges the sequence should be: st, est, ow, low, west, ne."""
    expected_merges = [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SENNRICH_CORPUS)
        tmp = f.name
    try:
        # 256 bytes + 1 special token + 6 merges = 263
        vocab, merges = run_train_bpe(
            input_path=tmp,
            vocab_size=263,
            special_tokens=[SPECIAL_TOKEN],
        )
        assert merges == expected_merges
    finally:
        os.unlink(tmp)


def test_sennrich_vocab_after_six_merges():
    """Vocab after 6 merges should contain the 256 byte tokens, the special token,
    and the 6 merged tokens: st, est, ow, low, west, ne."""
    expected_merged_tokens = {b"st", b"est", b"ow", b"low", b"west", b"ne"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SENNRICH_CORPUS)
        tmp = f.name
    try:
        vocab, merges = run_train_bpe(
            input_path=tmp,
            vocab_size=263,
            special_tokens=[SPECIAL_TOKEN],
        )
        assert len(vocab) == 263
        vocab_values = set(vocab.values())
        assert SPECIAL_TOKEN.encode("utf-8") in vocab_values
        assert expected_merged_tokens.issubset(vocab_values)
        # All 256 byte values present
        for i in range(256):
            assert bytes([i]) in vocab_values
    finally:
        os.unlink(tmp)


def test_sennrich_special_token_not_merged():
    """Special token should appear as a single vocab entry and never be split."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SENNRICH_CORPUS)
        tmp = f.name
    try:
        vocab, merges = run_train_bpe(
            input_path=tmp,
            vocab_size=263,
            special_tokens=[SPECIAL_TOKEN],
        )
        special_bytes = SPECIAL_TOKEN.encode("utf-8")
        # Appears exactly once in vocab values
        assert list(vocab.values()).count(special_bytes) == 1
        # Never appears as a component in any merge
        for left, right in merges:
            assert special_bytes not in (left, right)
    finally:
        os.unlink(tmp)


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently), we'll make sure that the vocab keys and values match
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )
