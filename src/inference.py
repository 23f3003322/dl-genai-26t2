import pickle
from pathlib import Path
import torch


from src.model import BiLSTMAttention
from src.preprocess import ScratchVocabulary, encode_mcq


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BILSTM_CFG = {

    "embedding_dim": 300,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "max_length": 128

}

ID2LABEL = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}

BASE_DIR = Path(__file__).resolve().parent.parent
VOCAB_PATH = BASE_DIR / "models" / "vocab.pkl"
MODEL_PATH = BASE_DIR / "models" / "best_bilstm_attention.pth"


class CompatVocabUnpickler(pickle.Unpickler):
    """Remap legacy notebook-pickled vocab class to module class."""

    def find_class(self, module, name):
        if module == "__main__" and name == "ScratchVocabulary":
            return ScratchVocabulary

        return super().find_class(module, name)


# ============================================================
# Load Vocabulary
# ============================================================

with VOCAB_PATH.open("rb") as f:

    vocab = CompatVocabUnpickler(f).load()


# ============================================================
# Load Model
# ============================================================

model = BiLSTMAttention(

    vocab_size=vocab.vocab_size,

    embedding_dim=BILSTM_CFG["embedding_dim"],

    hidden_dim=BILSTM_CFG["hidden_dim"],

    num_layers=BILSTM_CFG["num_layers"],

    dropout=BILSTM_CFG["dropout"]

)

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

)

model.to(DEVICE)

model.eval()


# ============================================================
# Prediction Function
# ============================================================

@torch.no_grad()
def predict(

    question,

    option_a,

    option_b,

    option_c,

    option_d,

    option_e

):

    inputs = encode_mcq(

        question,

        option_a,

        option_b,

        option_c,

        option_d,

        option_e,

        vocab=vocab,

        max_length=BILSTM_CFG["max_length"]

    )

    inputs = inputs.to(DEVICE)

    logits = model(inputs)

    probabilities = torch.softmax(
        logits,
        dim=1
    ).squeeze(0)

    probs = probabilities.cpu().numpy()

    prediction = probs.argmax()

    predicted_answer = ID2LABEL[prediction]

    probability_dict = {

        ID2LABEL[i]: float(probs[i])

        for i in range(5)

    }

    top3_indices = probs.argsort()[::-1][:3]

    top3 = [

        ID2LABEL[i]

        for i in top3_indices

    ]

    return (

        predicted_answer,

        " > ".join(top3),

        probability_dict

    )
