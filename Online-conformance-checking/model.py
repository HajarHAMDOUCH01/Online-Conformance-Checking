import torch
import torch.nn as nn
import torch.nn.functional as F

class PetriNetRotationLayer(nn.Module):
    def __init__(self,num_markings, num_transitions):
        super(PetriNetRotationLayer, self).__init__()
        self.num_markings = num_markings
        self.num_transitions = num_transitions
        
        """
        1. learnable generators for each transition (Lie algebra elements)
        for exp(generator) to be valid rotation, the geneerator matrix has to be antisymmetric
        an antisymmetric matrix : the diagonal is zeros and the bottom left is a negative mirror of the top right
        """

        # number of independent parameters in an antisymmetric matrix
        self.n_params = (num_markings * (num_markings - 1)) // 2  
        
        """
        parameters for the generators of each transition
        an embedding of the transition in the Lie algebra, which will be exponentiated to get the rotation matrix of the rotation between two markings
        """
        self.transition_generators = nn.Parameter(torch.randn(num_transitions, self.n_params))

        self.W = nn.Linear(num_markings, num_transitions)

    """the function that expresses each transtion in the structure of the reachability graph"""
    def full_generators(self):
        batch_gen = torch.zeros(self.num_transitions, self.num_markings, self.num_markings)
        tri_u_indices = torch.triu_indices(self.num_markings, self.num_markings, offset=1)
        batch_gen[:, tri_u_indices[0], tri_u_indices[1]] = self.transition_generators
        
        return batch_gen - batch_gen.transpose(1, 2)  # make it antisymmetric

    
    def get_rotation_matrix(self, alphas):
        """combines generators weighted by alpha and applies matrix exponential"""
        
        # create an antisymmetric empty matrix
        batch_size = alphas.size(0)
        
        # a temporary variable for calculating the composition of transitions (the shortest path of transitions to get an alignement)
        omega = torch.zeros(batch_size, self.num_markings, self.num_markings)
        combined_omega = torch.einsum('bt, tmk -> bmk', alphas, self.full_generators())
        return torch.matrix_exp(combined_omega)  # matrix exponential to get the rotation matrix

    def forward(self, v_current):
        alphas = self.W(v_current)  
        R_total = self.get_rotation_matrix(alphas)
        v_final = torch.bmm(R_total, v_current.unsqueeze(-1)).squeeze(-1)  # apply the rotation to the current marking
        return v_final, alphas
