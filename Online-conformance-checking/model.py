"""

Embedding the reachability graph of a Petri net into SO(n) 
where each marking is an axis of an n-dimensional sphere 
and using Lie algebra generators (antisymmetric matrices + matrix exponential) 
as the transition operators, trained with a 
neural network to do prefix-alignment for online conformance checking.

Stream: t1 → t2 → t_nonconformant → ...

1. Track current_marking after each conformant transition
2. Non-conformant transition t_x arrives
3. Look up in reachability_graph: "in which marking is t_x enabled?"
   → that marking is v_tgt
4. model(v_src=current_marking, v_tgt=found_marking) 
   → returns the conformant path to get there
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PetriNetAlignmentPredictor(nn.Module):
    def __init__(self, reachability_tensor):
        super().__init__()
        self.register_buffer('omegas', reachability_tensor)
        self.num_m = reachability_tensor.shape[1]
        self.num_t = reachability_tensor.shape[0]

        self.network_step = nn.Sequential(
            nn.Linear(self.num_m * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_t),
        )

    def forward(self, v_src, v_target, max_steps=20, training=True, tau=1.0):
        v_current = v_src       # [1, num_m]
        predicted_logits = []
        v_tgt_idx = v_target.argmax(dim=-1).item()

        for step in range(max_steps):
            x = torch.cat([v_current, v_target], dim=-1)  # [1, num_m*2]
            logits = self.network_step(x)                  # [1, num_t]

            if training:
                soft_one_hot = F.gumbel_softmax(logits, tau=1.0, hard=False)  # [1, num_t]
                omega_k = torch.einsum('bt,tmk->bmk', soft_one_hot, self.omegas)
            else:
                idx = logits.argmax(dim=-1)
                omega_k = self.omegas[idx]                 # [1, num_m, num_m]

            R_k = torch.matrix_exp(omega_k)
            v_current = torch.bmm(R_k, v_current.unsqueeze(-1)).squeeze(-1)

            predicted_logits.append(logits)

            if v_current.argmax(dim=-1).item() == v_tgt_idx:
                break

        full_logits_seq = torch.stack(predicted_logits, dim=1).squeeze(0)  # [L_pred, num_t]
        return v_current, full_logits_seq