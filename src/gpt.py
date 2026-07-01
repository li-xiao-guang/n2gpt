import numpy as np

from src.activation import Triu, Softmax, GeLU
from src.layer import Sequential, Embedding, Dropout, Normalize, Linear
from src.tensor import Tensor


class GPTEmbedding(Sequential):
    """
    Input representation: token embedding + positional embedding.

    GPT has no notion of position built into its attention mechanism,
    so position must be injected explicitly.  Both the token and the
    position are looked up in separate learned embedding tables and
    then summed — the model learns what positional information to
    encode during training.
    """

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
        # Positions are always 0, 1, …, T-1 — a simple integer range.
        position = self.positional_embedding(Tensor(range(seq_len)))
        return self.dropout(token + position)


class GPTAttention(Sequential):
    """
    Multi-head causal self-attention with pre-norm.

    PRE-NORM (LayerNorm before attention, not after):
      GPT-3 applies LayerNorm to the residual stream before each
      sub-layer (pre-norm), rather than after (post-norm as in the
      original "Attention is All You Need" paper).  Pre-norm produces
      more stable gradients in deep networks because the normalisation
      keeps activations well-scaled before they enter the attention
      computation.

    MULTI-HEAD SPLIT:
      Rather than one attention over the full embedding dimension C,
      we run H independent attentions over C/H dimensions each.
      Each head can specialise in a different type of relationship.
      Concretely we reshape (T, C) → (H, T, D) where D = C/H and
      run batched matrix operations.

    RESIDUAL CONNECTION:
      The output is added back to the original x (before the
      pre-norm).  This gives gradients a direct path back through
      the network and allows the layer to learn a residual correction
      rather than a full transformation — much easier to optimise.
    """

    def __init__(self, context_size, embedding_size, heads=1, dropout=0.1):
        assert embedding_size % heads == 0, "embedding_size must be divisible by heads"

        self.context_size = context_size
        self.embedding_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size // heads   # D = C / H

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
        norm = self.normalize(x)   # pre-norm

        # Project and split into H heads.
        # Linear maps (T, C) → (T, C), then reshape to (T, H, D),
        # then transpose to (H, T, D) for batched attention.
        query = self.attention_query(norm).reshape((seq_len, self.heads, self.head_dim)).transpose((1, 0, 2))
        key   = self.attention_key(norm).reshape((seq_len, self.heads, self.head_dim)).transpose((1, 0, 2))
        value = self.attention_value(norm).reshape((seq_len, self.heads, self.head_dim)).transpose((1, 0, 2))

        # Scaled dot-product attention.
        # (H, T, D) @ (H, D, T) → (H, T, T)
        # WHY scale by 1/√D?  Without scaling, for large D the dot
        # products grow large and push softmax into saturation regions
        # where gradients vanish.
        scale = Tensor(np.array(1.0 / np.sqrt(self.head_dim), dtype=query.data.dtype))
        scores = (query @ key.transpose((0, 2, 1))) * scale

        # Apply causal mask so position i cannot attend to j > i.
        scores = self.triu(scores)
        weights = self.softmax(scores)
        weights = self.attention_dropout(weights)

        # Weighted sum of values, then merge heads back into (T, C).
        # (H, T, T) @ (H, T, D) → (H, T, D) → (T, H, D) → (T, C)
        out = (weights @ value).transpose((1, 0, 2)).reshape((seq_len, self.embedding_size))
        out = self.output_dropout(self.merge(out))

        # Residual connection: input x bypasses the attention block.
        return x + out


class GPTFeedForward(Sequential):
    """
    Position-wise feed-forward network (FFN) with pre-norm.

    Each token position is processed independently by the same two
    linear layers with a GeLU activation in between.  The hidden
    size is 4× the embedding size — a design choice from the original
    Transformer paper that has been retained in GPT-3.

    Like attention, it uses:
      - Pre-norm: normalise the residual stream before the FFN.
      - Residual: add the FFN output back to x.
    """

    def __init__(self, embedding_size, ffn_hidden=None, dropout=0.1):
        self.embedding_size = embedding_size
        self.ffn_hidden = ffn_hidden if ffn_hidden is not None else embedding_size * 4

        self.normalize = Normalize(self.embedding_size)
        self.input = Linear(self.embedding_size, self.ffn_hidden)
        self.gelu = GeLU()
        self.output = Linear(self.ffn_hidden, self.embedding_size)
        self.dropout = Dropout(dropout)

        layers = [self.normalize, self.input, self.gelu, self.output, self.dropout]
        super().__init__(layers)

    def forward(self, x: Tensor):
        norm = self.normalize(x)
        hidden = self.gelu(self.input(norm))
        return x + self.dropout(self.output(hidden))   # residual


class GPTTransformer(Sequential):
    """
    One transformer block = attention sub-layer + FFN sub-layer.

    The block is the repeated unit that gets stacked N times.
    Increasing N makes the model deeper (more computation per token)
    while increasing embedding_size makes it wider (more capacity
    per layer).
    """

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
    """
    Final projection from the residual stream to vocabulary logits.

    LayerNorm is applied first (final pre-norm), then a linear layer
    maps (T, C) → (T, vocab_size).  The resulting logits are fed
    to the cross-entropy loss (which internally applies softmax).
    """

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
    """
    GPT-3 style decoder-only transformer.

    Architecture (top to bottom):
      GPTEmbedding         — token + positional embeddings
      GPTTransformer × N   — N stacked attention + FFN blocks
      GPTOutput            — final LayerNorm + linear to vocab

    WEIGHT TYING:
      The token embedding matrix and the final output linear layer
      share the same weight tensor.  Both have shape
      (vocabulary_size, embedding_size), so the assignment below
      works directly.

      Benefits:
        1. Saves vocabulary_size × embedding_size parameters.
        2. Forces the model to learn consistent input/output
           representations — the vector used to represent a token
           as input should also be a good target for predicting that
           token as output.

      During backprop, both the embedding's gradient_fn and the
      linear layer's gradient_fn write into the same tensor's .grad,
      so gradients from both sides accumulate automatically.

    PARAMETER DEDUPLICATION:
      Because the tied weight appears in two layers, a naive call to
      super().parameters() would return it twice.  The optimizer
      would then update it twice per step.  We deduplicate by
      object identity (id()) so each tensor is updated exactly once.
    """

    def __init__(self, vocabulary_size, context_size, embedding_size, heads=4, layers=4,
                 ffn_hidden=None, dropout=0.1):
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.heads = heads
        self.num_layers = layers

        self.embedding = GPTEmbedding(
            self.vocabulary_size, self.context_size, self.embedding_size, dropout=dropout
        )
        self.transformers = [
            GPTTransformer(
                self.context_size, self.embedding_size, self.heads,
                ffn_hidden=ffn_hidden, dropout=dropout
            )
            for _ in range(self.num_layers)
        ]
        self.output = GPTOutput(self.vocabulary_size, self.embedding_size)

        all_layers = [self.embedding] + self.transformers + [self.output]
        super().__init__(all_layers)

        # Weight tying: share the token embedding weight with the output projection.
        # Both tensors have shape (vocabulary_size, embedding_size) after our
        # Embedding convention change, so this assignment is shape-safe.
        self.output.output.weight = self.embedding.token_embedding.weight

    def parameters(self):
        # Deduplicate by object identity so the tied weight is only
        # returned once and the optimizer updates it exactly once per step.
        seen = set()
        unique = []
        for p in super().parameters():
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique

    def forward(self, x: Tensor):
        x = self.embedding(x)
        for layer in self.transformers:
            x = layer(x)
        return self.output(x)