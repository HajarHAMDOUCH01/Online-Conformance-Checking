"""
Scalable SO(n) alignment predictor

  Enabled  (src, t, dst):  R_t @ emb(src) ≈ emb(dst)
                            The rotation carries the marking to its valid successor.

  Disabled (m, t):         R_t @ emb(m)   ≈ emb(m)
                            The rotation acts as identity — firing a disabled
                            transition leaves the state unchanged.

Together these force R_t to have a specific eigenstructure:
  - Enabled markings  → non-trivial rotation toward their successor
  - Disabled markings → fixed point (identity eigenspace)

This means the model can never want to fire a disabled transition,
because doing so geometrically goes nowhere — no external masking needed.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PetriNetAlignmentPredictor(nn.Module):
    def __init__(self, reachability_tensor, n=16):
        super().__init__()
        self.num_t   = reachability_tensor.size(0)
        self.num_m   = reachability_tensor.size(1)
        self.n       = n
        self.num_gen = n * (n - 1) // 2

        # Marking embeddings: num_m points on S^{n-1}
        W = torch.randn(self.num_m, n)
        self.emb_weight = nn.Parameter(F.normalize(W, p=2, dim=-1))

        # SO(n) basis generators [num_gen, n, n]  (antisymmetric)
        bases = []
        for i in range(n):
            for j in range(i + 1, n):
                B = torch.zeros(n, n)
                B[i, j] = -1.0
                B[j, i] =  1.0
                bases.append(B)
        self.register_buffer('bases', torch.stack(bases))

        # Transition generator coefficients [num_t, num_gen]
        self.transition_coeffs = nn.Parameter(
            torch.randn(self.num_t, self.num_gen) * 0.1
        )

        # Policy network: [n*2] → [num_t]
        self.network_step = nn.Sequential(
            nn.Linear(n * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_t),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_embeddings(self):
        """Unit-normalised marking embeddings [num_m, n]"""
        return F.normalize(self.emb_weight, p=2, dim=-1)

    def encode(self, v_onehot):
        """[B, num_m] → [B, n]"""
        return v_onehot @ self.get_embeddings()

    def decode_scores(self, v_emb):
        """[B, n] → [B, num_m] cosine similarity scores"""
        return v_emb @ self.get_embeddings().T

    def get_generators(self):
        """[num_t, n, n]  (skew-symmetric)"""
        return torch.einsum('tg,gnm->tnm', self.transition_coeffs, self.bases)

    def _rotate(self, v_emb, logits, tau, training):
        generators = self.get_generators()
        if training:
            one_hot = F.gumbel_softmax(logits, tau=tau, hard=True)
            omega   = torch.einsum('bt,tnm->bnm', one_hot, generators)
        else:
            omega = generators[logits.argmax(dim=-1)]
        R     = torch.matrix_exp(omega)
        return F.normalize(
            torch.bmm(R, v_emb.unsqueeze(-1)).squeeze(-1), p=2, dim=-1
        )

    # ── Phase 1: geometry pre-training loss ───────────────────────────────────

    def geometry_loss(self, edge_list, disabled_pairs, lambda_disabled=1.0):
        """
        edge_list      : list of (src_idx, t_idx, dst_idx)  — enabled transitions
        disabled_pairs : list of (m_idx, t_idx)              — disabled transitions
        lambda_disabled: weight on the disabled term

        Enabled  loss: || R_t @ emb(src) - emb(dst) ||^2
        Disabled loss: || R_t @ emb(m)   - emb(m)   ||^2  =  || (R_t - I) @ emb(m) ||^2

        The two together force R_t to rotate enabled markings toward their successors
        while acting as identity on all disabled markings.
        """
        embs       = self.get_embeddings()    # [num_m, n]
        generators = self.get_generators()    # [num_t, n, n]
        dev        = self.emb_weight.device

        # ── Enabled term ──────────────────────────────────────────────────────
        src_idxs = torch.tensor([e[0] for e in edge_list], dtype=torch.long, device=dev)
        t_idxs   = torch.tensor([e[1] for e in edge_list], dtype=torch.long, device=dev)
        dst_idxs = torch.tensor([e[2] for e in edge_list], dtype=torch.long, device=dev)

        v_src  = embs[src_idxs]                                       # [E, n]
        v_dst  = embs[dst_idxs]                                       # [E, n]
        omegas = generators[t_idxs]                                   # [E, n, n]
        R_en   = torch.matrix_exp(omegas)                             # [E, n, n]
        v_rot  = F.normalize(
            torch.bmm(R_en, v_src.unsqueeze(-1)).squeeze(-1), p=2, dim=-1
        )
        loss_enabled = F.mse_loss(v_rot, v_dst)

        # ── Disabled term ─────────────────────────────────────────────────────
        dm_idxs = torch.tensor([p[0] for p in disabled_pairs], dtype=torch.long, device=dev)
        dt_idxs = torch.tensor([p[1] for p in disabled_pairs], dtype=torch.long, device=dev)

        v_m     = embs[dm_idxs]                                       # [D, n]
        omegas_d = generators[dt_idxs]                                # [D, n, n]
        R_dis   = torch.matrix_exp(omegas_d)                          # [D, n, n]
        v_rot_d = F.normalize(
            torch.bmm(R_dis, v_m.unsqueeze(-1)).squeeze(-1), p=2, dim=-1
        )
        loss_disabled = F.mse_loss(v_rot_d, v_m)   # target = stay in place

        return loss_enabled + lambda_disabled * loss_disabled

    def separation_loss(self, margin=0.1):
        """Push all marking embeddings apart — prevents collapse."""
        embs       = self.get_embeddings()
        sim_matrix = embs @ embs.T
        mask       = ~torch.eye(self.num_m, dtype=torch.bool, device=embs.device)
        violations = torch.clamp(sim_matrix[mask] - margin, min=0.0)
        return violations.mean()

    # ── Forward (policy mode) ─────────────────────────────────────────────────

    def forward(self, v_src, v_target, num_steps=None, training=True, tau=1.0):
        v_current = self.encode(v_src)
        v_tgt_emb = self.encode(v_target)
        v_tgt_idx = v_target.argmax(dim=-1).item()

        predicted_logits = []
        step_range = range(num_steps) if training else range(50)

        for _ in step_range:
            logits = self.network_step(
                torch.cat([v_current, v_tgt_emb], dim=-1)
            )
            predicted_logits.append(logits)
            v_current = self._rotate(v_current, logits, tau, training)

            if not training:
                if self.decode_scores(v_current).argmax(dim=-1).item() == v_tgt_idx:
                    break

        logits_seq   = torch.stack(predicted_logits, dim=1).squeeze(0)  # [L, num_t]
        pred_marking = F.softmax(self.decode_scores(v_current), dim=-1)  # [1, num_m]
        return pred_marking, logits_seq