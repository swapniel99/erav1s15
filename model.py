import torch
import torch.nn as nn
import math

'''
Two problems:
1. why layernorm doesnt have dimension
'''


class LayerNormalzation(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(LayerNormalzation, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x:  [batch_size, seq_len, hidden_size]
        # keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        std = x.std(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # x:  [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbeddings, self).__init__(vocab_size, d_model)
        self.sqrt_d_model = math.sqrt(d_model)

    def forward(self, x):
        # x:  [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        # Multiplies by sqrt(d_model) to scale embeddings according to the paper.
        # Done probably so that positional embeddings are not as loud as input embeddings
        return super(InputEmbeddings, self).forward(x) * self.sqrt_d_model


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Create a matrix of shape [seq_len, d_model]
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape [seq_len, 1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Create a vector of shape [d_model]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model)))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model)))
        # add a batch dimension to the positional embeddings to broadcast across batch
        # pe:  [seq_len, d_model] -> [1, seq_len, d_model]
        pe = pe.unsqueeze(0)
        # register the positional embedding as buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:  [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)  # [batch, seq_len, d_model]
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNormalzation()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, sublayer):
        # x:  [batch_size, seq_len, d_model]
        # sublayer:  [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        # norm is applied first for code simplicity. Have to check what does it simplify.
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads

        assert d_model % h == 0, "Embedding vector size 'd_model' must be divisible by number of heads 'h'"

        self.d_k = d_model // h  # Dimension of vector seen by each head

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # [batch_size, h, seq_len, d_k] -> [batch_size, h, seq_len, seq_len]
        attention_scores = (query @ key.transpose(-2, -1)) * (d_k ** -0.5)
        if mask is not None:
            # Write an extremely negative value indicating -inf to the positions where mask == 0
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # [batch_size, h, seq_len, seq_len]
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # [batch_size, h, seq_len, seq_len] -> [batch_size, h, seq_len, d_k]
        # also return attention scores which can be used for visualisation
        return attention_scores @ value

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # [batch_size, seq_len, d_model]
        key = self.w_k(k)    # [batch_size, seq_len, d_model]
        value = self.w_v(v)  # [batch_size, seq_len, d_model]

        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, h, d_k]-> [batch_size, h, seq_len, d_k]
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x = self.attention(query, key, value, mask, self.dropout)

        # Combine all heads
        # [batch_size, h, seq_len, d_k] -> [batch_size, seq_len, h, d_k] -> [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        # Multiply by Wo
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda inp: self.self_attention_block(inp, inp, inp, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalzation()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda inp: self.self_attention_block(inp, inp, inp, tgt_mask))
        x = self.residual_connections[1](x, lambda inp: self.cross_attention_block(inp, encoder_output, encoder_output,
                                                                                   src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalzation()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(ProjectionLayer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, vocab_size]
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEmbedding, tgt_pos: PositionalEmbedding, projection_layer: ProjectionLayer):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        src = self.src_embed(src)
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        src = self.src_pos(src)
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        tgt = self.tgt_embed(tgt)
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        tgt = self.tgt_pos(tgt)
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, vocab_size]
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(src_vocab_size, d_model)
    tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)

    # Create the positional embedding layers
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = list()
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = list()
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,
                                     decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters. Optional since pytorch does this anyways.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
