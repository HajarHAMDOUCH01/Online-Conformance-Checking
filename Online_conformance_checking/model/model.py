"""
model.py

Transformer Encoder-Decoder for online conformance checking.

Architecture
------------
Encoder:
    - Activity embedding + positional encoding
    - N Transformer encoder layers
    - CLS token pooling → latent vector z
    - L2 normalization of z (for contrastive loss on hypersphere)

Decoder:
    - Activity embedding + positional encoding  (on conforming prefix)
    - N Transformer decoder layers
        * Masked causal self-attention  (can't look at future tokens)
        * Cross-attention over z        (broadcast as seq of length 1)
    - Linear projection → vocab logits (causal next-activity prediction)

Losses (computed outside the model, guided by forward outputs):
    - Reconstruction : cross-entropy over decoder logits vs conforming prefix
    - Contrastive    : NT-Xent on (z_noisy, z_conforming) pairs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds position information to token embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                        # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                       # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class PrefixEncoder(nn.Module):
    """
    Encodes a prefix (conforming or noisy) into a single latent vector z.

    A learned CLS token is prepended to the sequence.  After the Transformer
    encoder layers the CLS position is extracted and L2-normalised to live on
    the unit hypersphere — a requirement for the NT-Xent contrastive loss.

    Parameters
    ----------
    vocab_size   : total number of activity tokens (including PAD / UNK)
    d_model      : embedding / hidden dimension
    nhead        : number of attention heads  (must divide d_model)
    num_layers   : number of Transformer encoder layers
    dim_feedforward: inner dimension of the FFN inside each encoder layer
    dropout      : dropout probability
    pad_idx      : index of the <PAD> token (no gradient flows through it)
    max_len      : maximum prefix length supported by positional encoding
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 512,
    ):
        super().__init__()

        self.d_model  = d_model
        self.pad_idx  = pad_idx

        # ── token embedding ──────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # ── learned CLS token ────────────────────────────────────────────
        # prepended to every sequence; its final hidden state becomes z
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ── positional encoding ──────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len + 1)
                                                            # +1 for CLS position

        # ── transformer encoder ──────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,       # input shape: [B, seq, d_model]
            norm_first=True,        # pre-norm → more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,            # [B, seq_len]  integer activity indices
        src_key_padding_mask: torch.Tensor | None = None,
                                    # [B, seq_len]  True where PAD
    ) -> torch.Tensor:
        """
        Returns
        -------
        z : [B, d_model]  L2-normalised latent vector (on unit hypersphere)
        """
        B, S = x.shape

        # 1. embed activities
        emb = self.embedding(x) * math.sqrt(self.d_model)   # [B, S, d_model]

        # 2. prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)               # [B, 1, d_model]
        emb = torch.cat([cls, emb], dim=1)                   # [B, S+1, d_model]

        # 3. positional encoding
        emb = self.pos_encoding(emb)                         # [B, S+1, d_model]

        # 4. extend padding mask to account for the prepended CLS position
        #    CLS is never masked (False = attend)
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
                                                             # [B, S+1]

        # 5. transformer encoder
        enc_out = self.transformer_encoder(
            emb,
            src_key_padding_mask=src_key_padding_mask,
        )                                                    # [B, S+1, d_model]

        # 6. extract CLS position → pooled representation
        cls_out = enc_out[:, 0, :]                           # [B, d_model]

        # 7. L2 normalise → unit hypersphere (required for NT-Xent)
        z = F.normalize(cls_out, p=2, dim=-1)                # [B, d_model]

        return z


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class PrefixDecoder(nn.Module):
    """
    Autoregressively reconstructs the conforming prefix from latent vector z.

    The latent vector z is broadcast as a memory sequence of length 1.
    Each decoder layer cross-attends to this memory, conditioning every
    generated token on the full prefix encoding.

    During training teacher-forcing is used: the full conforming prefix is
    fed as input shifted right (prepend BOS), and a causal mask prevents
    each position from attending to future positions.

    Parameters
    ----------
    vocab_size     : total number of activity tokens
    d_model        : must match PrefixEncoder d_model
    nhead          : number of attention heads
    num_layers     : number of Transformer decoder layers
    dim_feedforward: inner FFN dimension
    dropout        : dropout probability
    pad_idx        : <PAD> token index
    bos_idx        : begin-of-sequence token index (we reuse pad=0 by default;
                     set to a dedicated index if you add a BOS token to vocab)
    max_len        : maximum sequence length
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bos_idx: int = 0,
        max_len: int = 512,
    ):
        super().__init__()

        self.d_model   = d_model
        self.pad_idx   = pad_idx
        self.bos_idx   = bos_idx
        self.vocab_size = vocab_size

        # ── token embedding ──────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # ── positional encoding ──────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        # ── project z [d_model] → memory sequence [1, d_model] ──────────
        # A simple linear keeps z in the same space; the decoder cross-attends
        # to this single-token memory at every layer.
        self.z_projection = nn.Linear(d_model, d_model)

        # ── transformer decoder ──────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── output projection → vocabulary logits ────────────────────────
        self.output_projection = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular causal mask.
        Shape [seq_len, seq_len], True = position is masked (ignored).
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,                        # [B, d_model]  from encoder
        tgt: torch.Tensor,                      # [B, tgt_len]  conforming prefix (teacher-forced)
        tgt_key_padding_mask: torch.Tensor | None = None,
                                                # [B, tgt_len]  True where PAD
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : [B, tgt_len, vocab_size]
            Raw (un-normalised) scores for each position.
            Apply cross-entropy against the target shifted by one position.
        """
        B, T = tgt.shape

        # 1. embed target sequence
        emb = self.embedding(tgt) * math.sqrt(self.d_model)  # [B, T, d_model]
        emb = self.pos_encoding(emb)                         # [B, T, d_model]

        # 2. causal self-attention mask
        causal_mask = self._causal_mask(T, tgt.device)       # [T, T]

        # 3. prepare memory from z  →  [B, 1, d_model]
        memory = self.z_projection(z).unsqueeze(1)           # [B, 1, d_model]

        # 4. transformer decoder
        dec_out = self.transformer_decoder(
            tgt=emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,       # memory length is always 1, never padded
        )                                                    # [B, T, d_model]

        # 5. project to vocabulary
        logits = self.output_projection(dec_out)             # [B, T, vocab_size]

        return logits

    # ------------------------------------------------------------------
    # Inference utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,            # [B, d_model]
        max_len: int = 50,
        eos_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation given latent vector z.
        Starts with BOS token and generates until max_len or eos_idx.

        Returns
        -------
        generated : [B, generated_len]  integer activity indices
        """
        B = z.size(0)
        device = z.device

        generated = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(z, generated)          # [B, cur_len, vocab_size]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                                                         # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

            # stop if all sequences produced EOS
            if eos_idx is not None and (next_token == eos_idx).all():
                break

        return generated[:, 1:]   # strip BOS


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class PrefixConformanceModel(nn.Module):
    """
    Full encoder-decoder model for prefix conformance checking.

    Forward pass returns:
        z_noisy      : [B, d_model]  latent vector of noisy prefix  (L2-normed)
        z_conforming : [B, d_model]  latent vector of conforming prefix (L2-normed)
        logits       : [B, tgt_len, vocab_size]  decoder output
                       trained to reconstruct the conforming prefix
                       from the NOISY prefix encoding z_noisy

    Loss signals (computed in training loop):
        Reconstruction  : cross_entropy(logits, conforming_prefix shifted by 1)
        Contrastive     : NT-Xent(z_noisy, z_conforming)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bos_idx: int = 0,
        max_len: int = 512,
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.encoder = PrefixEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_idx=pad_idx,
            max_len=max_len,
        )

        self.decoder = PrefixDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            max_len=max_len,
        )

    # ------------------------------------------------------------------

    def _padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """True where x == pad_idx  →  [B, seq_len]"""
        return x == self.pad_idx

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode any prefix (noisy or conforming) to a latent vector.

        Parameters
        ----------
        x : [B, seq_len]  integer activity indices

        Returns
        -------
        z : [B, d_model]  L2-normalised latent vector
        """
        mask = self._padding_mask(x)
        return self.encoder(x, src_key_padding_mask=mask)

    # ------------------------------------------------------------------

    def forward(
        self,
        noisy: torch.Tensor,        # [B, noisy_len]   noisy prefix
        conforming: torch.Tensor,   # [B, conf_len]    conforming prefix
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_noisy      : [B, d_model]
        z_conforming : [B, d_model]
        logits       : [B, conf_len, vocab_size]
            Decoder predicts conforming prefix token by token,
            conditioned on z_noisy (the noisy prefix encoding).
            This forces the encoder to map noisy → same z as conforming.
        """
        # ── encode both sides ────────────────────────────────────────────
        z_noisy      = self.encode(noisy)       # [B, d_model]
        z_conforming = self.encode(conforming)  # [B, d_model]

        # ── decode conforming prefix from noisy encoding ─────────────────
        # teacher forcing: feed full conforming prefix as decoder input
        # logits at position t predict token at position t+1
        tgt_mask = self._padding_mask(conforming)
        logits = self.decoder(
            z=z_noisy,
            tgt=conforming,
            tgt_key_padding_mask=tgt_mask,
        )                                       # [B, conf_len, vocab_size]

        return z_noisy, z_conforming, logits

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def align(
        self,
        noisy: torch.Tensor,        # [B, noisy_len]
        max_len: int = 50,
        eos_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Given a noisy (possibly non-conforming) prefix, encode it and
        autoregressively generate the aligned conforming prefix.

        Returns
        -------
        aligned : [B, generated_len]  predicted conforming activity sequence
        """
        z = self.encode(noisy)
        return self.decoder.generate(z, max_len=max_len, eos_idx=eos_idx)

    # ------------------------------------------------------------------
    # Conformance score
    # ------------------------------------------------------------------

    @torch.no_grad()
    def conformance_score(
        self,
        noisy: torch.Tensor,        # [B, noisy_len]
        conforming: torch.Tensor,   # [B, conf_len]   reference conforming prefix
    ) -> torch.Tensor:
        """
        Cosine similarity between z_noisy and z_conforming in latent space.
        Score ∈ [-1, 1].  Higher = more conforming.

        Since both vectors are already L2-normalised, this is just a dot product.
        """
        z_n = self.encode(noisy)
        z_c = self.encode(conforming)
        return (z_n * z_c).sum(dim=-1)          # [B]


# ---------------------------------------------------------------------------
# NT-Xent contrastive loss (used in training loop)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy loss (SimCLR).

    Expects z1 and z2 to already be L2-normalised (which the encoder ensures).

    For a batch of N pairs:
        - 2N embeddings total
        - For each anchor the positive is its pair, negatives are all others
        - Loss is symmetric: computed from both z1→z2 and z2→z1 directions
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,   # [B, d_model]  e.g. z_noisy
        z2: torch.Tensor,   # [B, d_model]  e.g. z_conforming
    ) -> torch.Tensor:
        B = z1.size(0)
        device = z1.device

        # concatenate both views → [2B, d_model]
        z = torch.cat([z1, z2], dim=0)

        # cosine similarity matrix [2B, 2B]
        # z is already L2-normed so dot product = cosine similarity
        sim = torch.mm(z, z.T) / self.temperature

        # mask out self-similarity on the diagonal
        mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))

        # positive pairs:
        #   for index i in [0, B)    the positive is at index i+B
        #   for index i in [B, 2B)   the positive is at index i-B
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B,     device=device),
        ])                                          # [2B]

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VOCAB_SIZE = 13
    D_MODEL    = 128
    BATCH      = 4
    PAD_IDX    = 0

    model = PrefixConformanceModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        pad_idx=PAD_IDX,
    )

    contrastive_loss_fn = NTXentLoss(temperature=0.07)

    # fake batch: variable-length prefixes padded to same length
    noisy      = torch.randint(1, VOCAB_SIZE, (BATCH, 7))
    conforming = torch.randint(1, VOCAB_SIZE, (BATCH, 6))

    z_noisy, z_conf, logits = model(noisy, conforming)

    print(f"z_noisy shape      : {z_noisy.shape}")        # [4, 128]
    print(f"z_conforming shape : {z_conf.shape}")         # [4, 128]
    print(f"logits shape       : {logits.shape}")         # [4, 6, 13]

    # reconstruction loss: predict next token → shift target by 1
    # input:  conforming[:, :-1]   (all but last)
    # target: conforming[:, 1:]    (all but first)
    recon_input  = conforming[:, :-1]
    recon_target = conforming[:, 1:]
    _, _, logits_shifted = model(noisy, recon_input)
    recon_loss = F.cross_entropy(
        logits_shifted.reshape(-1, VOCAB_SIZE),
        recon_target.reshape(-1),
        ignore_index=PAD_IDX,
    )

    # contrastive loss
    cont_loss = contrastive_loss_fn(z_noisy, z_conf)

    total_loss = recon_loss + 0.5 * cont_loss

    print(f"recon_loss         : {recon_loss.item():.4f}")
    print(f"contrastive_loss   : {cont_loss.item():.4f}")
    print(f"total_loss         : {total_loss.item():.4f}")

    # alignment inference
    aligned = model.align(noisy, max_len=10)
    print(f"aligned shape      : {aligned.shape}")        # [4, <=10]

    # conformance score
    score = model.conformance_score(noisy, conforming)
    print(f"conformance scores : {score}")                # [4]

    print("\nSmoke test passed.")