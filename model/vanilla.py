import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(d_model))  # bias is a learnable parameter

    def forward(self, x):
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        # (batch, seq_len, d_model)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.linear_1(x).relu()))


class InputEmbeddings(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(InputEmbeddings, self).__init__(vocab_size, d_model)
        self.sqrt_d_model = math.sqrt(d_model)

        for p in self.parameters():
            nn.init.xavier_uniform_(p)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        # Done probably so that positional embeddings are not as loud as input embeddings
        return super(InputEmbeddings, self).forward(x) * self.sqrt_d_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # Create a vector of shape (max_seq_len)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))  # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model)))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model)))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f"Expected sequence length {x.shape[1]} <= {self.max_seq_len} in Position Encoder.")
        # (batch, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        # (batch, seq_len, d_model)
        # return self.norm(x + self.dropout(sublayer(x)))  # Not working with adjustment in enc and dec.
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.parameters():
            nn.init.xavier_uniform_(p)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len1, d_k) @ (batch, h, d_k, seq_len2) --> (batch, h, seq_len1, seq_len2)
        attention_scores = (query @ key.transpose(-2, -1)) * (d_k ** -0.5)
        del query, key
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            _MASKING_VALUE = -3e4 if attention_scores.dtype == torch.float16 else -2e9
            attention_scores.masked_fill_(mask == 0, _MASKING_VALUE)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len1, seq_len2) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len1, seq_len2) @ (batch, h, seq_len2, d_k)--> (batch, h, seq_len1, d_k)
        # return attention scores which can be used for visualization
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask, return_attention=False):
        query = self.w_q(q)  # (batch, seq_len1, d_model)
        key = self.w_k(k)  # (batch, seq_len2, d_model)
        value = self.w_v(v)  # (batch, seq_len2, d_model)

        del q, k, v

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len1, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len2, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len2, d_k)

        # Calculate attention
        x, attention_scores = self.attention(query, key, value, mask, self.dropout)  # (batch, h, seq_len1, d_k), (batch, h, seq_len1, seq_len2)

        del query, key, value

        # Combine all the heads together
        # (batch, h, seq_len1, d_k) --> (batch, seq_len1, h, d_k) --> (batch, seq_len1, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)  # d_model = h * d_k

        # Multiply by Wo
        # (batch, seq_len1, d_model)
        x = self.w_o(x)
        if return_attention:
            return x, attention_scores
        else:
            return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList(ResidualConnection(d_model, dropout) for _ in range(2))

    def forward(self, x, src_mask):
        # x: (batch, e_seq_len, d_model)
        # src_mask: (batch, 1, 1, e_seq_len)
        x = self.residual_connections[0](x, lambda inp: self.self_attention_block(inp, inp, inp, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float) -> None:
        super(DecoderBlock, self).__init__()
        self.self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        self.cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList(ResidualConnection(d_model, dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x: (batch, d_seq_len, d_model)
        # encoder_output: (batch, e_seq_len, d_model)
        # src_mask: (batch, 1, 1, e_seq_len)
        # tgt_mask: (batch, 1, d_seq_len, d_seq_len)
        x = self.residual_connections[0](x, lambda inp: self.self_attention_block(inp, inp, inp, tgt_mask))
        x = self.residual_connections[1](x, lambda inp: self.cross_attention_block(inp, encoder_output, encoder_output,
                                                                                   src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class XCoder(nn.Module):
    def __init__(self, XcoderBlock: type, d_model: int, N: int, h: int, d_ff: int, dropout: float, param_sharing=None,
                 first_norm=True) -> None:
        super(XCoder, self).__init__()
        self.norm = LayerNormalization(d_model) if first_norm else nn.Identity()

        if param_sharing is None:
            self.layers = nn.ModuleList(XcoderBlock(d_model, h, d_ff, dropout) for _ in range(N))
        else:
            if N % 2 == 1:
                print("WARNING: Cannot share parameters in odd number of layers.")
            M = N // 2

            temp_layers = [XcoderBlock(d_model, h, d_ff, dropout) for _ in range(M)]
            self.layers = nn.ModuleList()

            if param_sharing == 'sequence':
                for layer in temp_layers:
                    for _ in range(2):
                        self.layers.append(layer)
            elif param_sharing == 'cycle':
                for _ in range(2):
                    for layer in temp_layers:
                        self.layers.append(layer)
            elif param_sharing == 'cycle_rev':
                for layer in temp_layers:
                    self.layers.append(layer)
                for layer in reversed(temp_layers):
                    self.layers.append(layer)
            else:
                raise ValueError(f'Unknown parameter sharing {param_sharing}')


class Encoder(XCoder):
    def __init__(self, d_model: int, N: int, h: int, d_ff: int, dropout: float, param_sharing=None,
                 first_norm=True) -> None:
        super(Encoder, self).__init__(EncoderBlock, d_model, N, h, d_ff, dropout, param_sharing, first_norm)

    def forward(self, x, src_mask):
        # x: (batch, e_seq_len, d_model)
        # src_mask: (batch, 1, 1, e_seq_len)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(XCoder):
    def __init__(self, d_model: int, N: int, h: int, d_ff: int, dropout: float, param_sharing=None,
                 first_norm=True) -> None:
        super(Decoder, self).__init__(DecoderBlock, d_model, N, h, d_ff, dropout, param_sharing, first_norm)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x: (batch, d_seq_len, d_model)
        # encoder_output: (batch, e_seq_len, d_model)
        # src_mask: (batch, 1, 1, e_seq_len)
        # tgt_mask: (batch, 1, d_seq_len, d_seq_len)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, tgt_vocab_size: int) -> None:
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(d_model, tgt_vocab_size)

        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        # (batch, d_seq_len, d_model) --> (batch, d_seq_len, tgt_vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, max_seq_len: int = 350, d_model: int = 512, N: int = 6,
                 heads: int = 8, dropout: float = 0.1, d_ff: int = 2048, param_sharing=None) -> None:
        """
        :param src_vocab_size: Source Vocab Size
        :param tgt_vocab_size: Target Vocab Size
        :param max_seq_len: Maximum Sequence Length
        :param d_model: Dimensionality of the model. Default is 512.
        :param N: Number of Encoder Blocks. Default is 6.
        :param heads: Number of Heads. Default is 8.
        :param dropout: Dropout Rate. Default is 0.1.
        :param d_ff: Dimensionality of the Feed Forward Network. Default is 2048.
        :return: None.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, N, heads, d_ff, dropout, param_sharing)
        self.decoder = Decoder(d_model, N, heads, d_ff, dropout, param_sharing)
        self.src_embed = InputEmbeddings(src_vocab_size, d_model)
        self.tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_seq_len, dropout)
        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        # src: (batch, e_seq_len)
        # src_mask: (batch, 1, 1, e_seq_len)
        src = self.src_embed(src)
        src = self.pos_embed(src)
        return self.encoder(src, src_mask)  # (batch, e_seq_len, d_model)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # encoder_output: (batch, e_seq_len, d_model)
        # src_mask: (batch, 1, 1, e_seq_len)
        # tgt: (batch, d_seq_len)
        # tgt_mask: (batch, 1, d_seq_len, d_seq_len)
        tgt = self.tgt_embed(tgt)
        tgt = self.pos_embed(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # (batch, d_seq_len, d_model)

    def project(self, x):
        # (batch, d_seq_len, d_model) --> (batch, d_seq_len, tgt_vocab_size)
        return self.projection_layer(x)

    def forward(self, src, src_mask, tgt, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        del src
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        del encoder_output, src_mask, tgt, tgt_mask
        return self.project(decoder_output)
