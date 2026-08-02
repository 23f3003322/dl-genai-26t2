import re
from collections import Counter
import torch


def clean_text(text):
    """
    Simple text normalization for the scratch model.
    """

    text = str(text).lower()

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def tokenizer(text):
    """
    Simple scratch tokenizer.
    Lowercases text and extracts words, numbers and <SEP>.
    """

    text = text.lower()

    return re.findall(r"<sep>|[a-z0-9']+", text)



class ScratchVocabulary:

    def __init__(self, min_freq=2):

        self.min_freq = min_freq

        self.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SEP>": 2
        }

        self.idx2word = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<SEP>"
        }

        self.vocab_size = 3


    def numericalize(self, text):

        tokens = tokenizer(text)

        return [
            self.word2idx.get(token, self.word2idx["<UNK>"])
            for token in tokens
        ]


def pad_sequence(token_ids, vocab, max_length):

    if len(token_ids) > max_length:

        token_ids = token_ids[:max_length]

    else:

        token_ids = token_ids + \
            [vocab.word2idx["<PAD>"]] * \
            (max_length - len(token_ids))

    return token_ids

def encode_mcq(
    prompt,
    option_a,
    option_b,
    option_c,
    option_d,
    option_e,
    vocab,
    max_length
):

    prompt = clean_text(prompt)

    option_sequences = []

    options = [
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    ]

    for option in options:

        option_text = clean_text(option)

        combined = (
            prompt
            + " <SEP> "
            + option_text
        )

        token_ids = vocab.numericalize(
            combined
        )

        token_ids = pad_sequence(
            token_ids,
            vocab,
            max_length
        )

        option_sequences.append(token_ids)

    option_sequences = torch.tensor(
        option_sequences,
        dtype=torch.long
    )

    # Add batch dimension
    return option_sequences.unsqueeze(0)


