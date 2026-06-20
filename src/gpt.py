import numpy as np

from src.activation import Triu, Softmax, ReLU
from src.layer import Sequential, Embedding, Dropout, Normalize, Linear
from src.tensor import Tensor


class GPTEmbedding(Sequential):

    def __init__(self, vocabulary_size, context_size, embedding_size, dropout=0.1):
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
        self.embedding_size = embedding_size

        self.token_embedding = Embedding(self.vocabulary_size, self.embedding_size)
        self.positional_embedding = Embedding(self.context_size, self.embedding_size)
        self.dropout = Dropout(dropout)

        layers = [self.token_embedding, self.positional_embedding, self.dropout]
        super().__init__(layers)

    def forward(self, x: Tensor):
        seq_len = len(x.data)
        assert seq_len <= self.context_size, (
            f"sequence length {seq_len} exceeds context_size {self.context_size}"
        )

        token = self.token_embedding(x)
        position = self.positional_embedding(Tensor(range(seq_len)))
        return self.dropout(token + position)


class GPTAttention(Sequential):

    def __init__(self, context_size, embedding_size, heads=1, dropout=0.1):
        assert embedding_size % heads == 0, "embedding_size must be divisible by heads"

        self.context_size = context_size
        self.embedding_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size // heads

        self.normalize = Normalize(self.embedding_size)
        self.attention_query = Linear(self.embedding_size, self.embedding_size)
        self.attention_key = Linear(self.embedding_size, self.embedding_size)
        self.attention_value = Linear(self.embedding_size, self.embedding_size)
        self.triu = Triu()
        self.softmax = Softmax()
        self.attention_dropout = Dropout(dropout)
        self.merge = Linear(self.embedding_size, self.embedding_size)
        self.output_dropout = Dropout(dropout)

        layers = [
            self.normalize,
            self.attention_query,
            self.attention_key,
            self.attention_value,
            self.triu,
            self.softmax,
            self.attention_dropout,
            self.merge,
            self.output_dropout,
        ]
        super().__init__(layers)

    def forward(self, x: Tensor):
        seq_len = x.data.shape[0]
        norm = self.normalize(x)

        # (T, C) -> (T, H, D) -> (H, T, D)
        query = self.attention_query(norm).reshape((seq_len, self.heads, self.head_dim)).transpose((1, 0, 2))
        key = self.attention_key(norm).reshape((seq_len, self.heads, self.head_dim)).transpose((1, 0, 2))
        value = self.attention_value(norm).reshape((seq_len, self.heads, self.head_dim)).transpose((1, 0, 2))

        # (H, T, D) @ (H, D, T) -> (H, T, T)
        scale = Tensor(np.array(1.0 / np.sqrt(self.head_dim), dtype=query.data.dtype))
        scores = (query @ key.transpose((0, 2, 1))) * scale
        scores = self.triu(scores)
        weights = self.softmax(scores)
        weights = self.attention_dropout(weights)

        # (H, T, T) @ (H, T, D) -> (H, T, D) -> (T, H, D) -> (T, C)
        out = (weights @ value).transpose((1, 0, 2)).reshape((seq_len, self.embedding_size))
        out = self.output_dropout(self.merge(out))
        return x + out


class GPTFeedForward(Sequential):

    def __init__(self, embedding_size, ffn_hidden=None, dropout=0.1):
        self.embedding_size = embedding_size
        self.ffn_hidden = ffn_hidden if ffn_hidden is not None else embedding_size * 4

        self.normalize = Normalize(self.embedding_size)
        self.input = Linear(self.embedding_size, self.ffn_hidden)
        self.relu = ReLU()
        self.output = Linear(self.ffn_hidden, self.embedding_size)
        self.dropout = Dropout(dropout)

        layers = [self.normalize, self.input, self.relu, self.output, self.dropout]
        super().__init__(layers)

    def forward(self, x: Tensor):
        norm = self.normalize(x)
        hidden = self.relu(self.input(norm))
        return x + self.dropout(self.output(hidden))


class GPTTransformer(Sequential):

    def __init__(self, context_size, embedding_size, heads, ffn_hidden=None, dropout=0.1):
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.heads = heads

        self.attention = GPTAttention(self.context_size, self.embedding_size, self.heads, dropout=dropout)
        self.feed_forward = GPTFeedForward(self.embedding_size, ffn_hidden=ffn_hidden, dropout=dropout)

        layers = [self.attention, self.feed_forward]
        super().__init__(layers)

    def forward(self, x: Tensor):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class GPTOutput(Sequential):

    def __init__(self, vocabulary_size, embedding_size):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.normalize = Normalize(self.embedding_size)
        self.output = Linear(self.embedding_size, self.vocabulary_size)

        layers = [self.normalize, self.output]
        super().__init__(layers)

    def forward(self, x: Tensor):
        norm = self.normalize(x)
        return self.output(norm)


class GPT(Sequential):

    def __init__(self, vocabulary_size, context_size, embedding_size, heads=4, layers=4,
                 ffn_hidden=None, dropout=0.1):
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.heads = heads
        self.num_layers = layers

        self.embedding = GPTEmbedding(self.vocabulary_size, self.context_size, self.embedding_size, dropout=dropout)
        self.transformers = [
            GPTTransformer(self.context_size, self.embedding_size, self.heads, ffn_hidden=ffn_hidden, dropout=dropout)
            for _ in range(self.num_layers)
        ]
        self.output = GPTOutput(self.vocabulary_size, self.embedding_size)

        all_layers = [self.embedding] + self.transformers + [self.output]
        super().__init__(all_layers)

    def forward(self, x: Tensor):
        x = self.embedding(x)
        for layer in self.transformers:
            x = layer(x)
        return self.output(x)
