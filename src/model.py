import torch
import torch.nn as nn


class BiLSTMAttention(nn.Module):
    """
    BiLSTM + Attention model for Multiple Choice Question Answering.

    Input Shape:
        [batch_size, 5, sequence_length]

    Output Shape:
        [batch_size, 5]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        padding_idx: int = 0
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # --------------------------------------------------
        # Attention Network
        # --------------------------------------------------

        self.attention = nn.Sequential(

            nn.Linear(
                lstm_output_dim,
                lstm_output_dim
            ),

            nn.Tanh(),

            nn.Linear(
                lstm_output_dim,
                1
            )

        )

        # --------------------------------------------------
        # Classification Head
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(
                lstm_output_dim,
                lstm_output_dim
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                lstm_output_dim,
                1
            )

        )

        self._initialize_weights()

    # ======================================================
    # Weight Initialization
    # ======================================================

    def _initialize_weights(self):

        for name, param in self.named_parameters():

            if "weight" in name:

                if param.dim() > 1:

                    nn.init.xavier_uniform_(param)

            elif "bias" in name:

                nn.init.zeros_(param)

    # ======================================================
    # Forward Pass
    # ======================================================

    def forward(self, x):

        """
        Parameters
        ----------
        x : Tensor

            Shape:
                [batch_size, 5, sequence_length]

        Returns
        -------
        logits : Tensor

            Shape:
                [batch_size, 5]
        """

        batch_size, num_options, sequence_length = x.size()

        # Merge option dimension into batch

        x = x.view(
            batch_size * num_options,
            sequence_length
        )

        embedded = self.embedding(x)

        lstm_output, _ = self.lstm(embedded)

        attention_scores = self.attention(
            lstm_output
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context_vector = torch.sum(

            attention_weights * lstm_output,

            dim=1

        )

        logits = self.classifier(
            context_vector
        )

        logits = logits.view(
            batch_size,
            num_options
        )

        return logits
