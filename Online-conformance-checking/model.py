import torch
import torch.nn as nn
import torch.nn.functional as F

class PetriNetAlignmentPredictor(nn.Module):
    def __init__(self, reachability_tensor):
        super().__init__()
        self.register_buffer('omegas', reachability_tensor)
        num_m = reachability_tensor.shape[1]
        num_t = reachability_tensor.shape[0]

        # On concatène v_source et v_target en entrée
        self.network = nn.Sequential(
            nn.Linear(num_m * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_t),
            nn.Sigmoid()
        )

    def forward(self, v_current, v_target):
        # On regarde les deux marquages pour décider du chemin
        x = torch.cat([v_current, v_target], dim=-1)
        alphas = self.network(x)
        
        # Calcul de la rotation de Lie
        combined_omega = torch.einsum('bt, tmk -> bmk', alphas, self.omegas)
        R = torch.matrix_exp(combined_omega)
        
        v_final = torch.bmm(R, v_current.unsqueeze(-1)).squeeze(-1)
        return v_final, alphas
