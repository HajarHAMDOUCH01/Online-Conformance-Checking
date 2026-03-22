import torch.optim as optim
import torch
import torch.nn.functional as F
from model import PetriNetAlignmentPredictor
from reachability_graph import *
from dataset import *
from train import *

def online_conformance_check(event_stream, model, reachability_graph, 
                              transition_to_enabled_markings, marking_to_idx):
    """
    event_stream: list of transition names, e.g. ['t1', 't2', 't5', 't3', ...]
    """
    current_marking_idx = 0  
    prefix_alignment = []
    
    for t_name in event_stream:
        current_marking = all_markings[current_marking_idx]
        
        enabled_transitions = [t for t, _, _ in reachability_graph.get(current_marking, [])]
        
        if t_name in enabled_transitions:
            for t, mode, next_m in reachability_graph[current_marking]:
                if t == t_name:
                    current_marking_idx = marking_to_idx[next_m]
                    break
            prefix_alignment.append((t_name, 'SYNC'))
            print(f"{t_name}  conformant, now at marking {current_marking_idx}")
        
        else:
            if t_name not in transition_to_enabled_markings:
                print(f"{t_name}  unknown transition, skip")
                prefix_alignment.append((t_name, 'SKIP'))
                continue
            
            # go the marking where t_name is enabled (v_tgt)
            candidates = transition_to_enabled_markings[t_name]
            
            best_path = None
            best_tgt_idx = None
            best_len = float('inf')
            
            for candidate in candidates:
                v_tgt_idx = candidate['from_idx']
                
                if v_tgt_idx == current_marking_idx:
                    continue
                
                v_src_tensor = torch.zeros(num_m); v_src_tensor[current_marking_idx] = 1.0
                v_tgt_tensor = torch.zeros(num_m); v_tgt_tensor[v_tgt_idx] = 1.0
                
                with torch.no_grad():
                    v_pred, pred_logits = model(
                        v_src_tensor.unsqueeze(0), 
                        v_tgt_tensor.unsqueeze(0), 
                        training=False
                    )
                
                path_len = pred_logits.size(0)
                if path_len < best_len:
                    best_len = path_len
                    best_path = pred_logits
                    best_tgt_idx = candidate['to_idx']  
            
            if best_path is not None:
                corrective_steps = []
                for step in range(best_path.size(0)):
                    probs = F.softmax(best_path[step], dim=-1)
                    top_t_idx = probs.argmax().item()
                    top_t_name = all_transition_names[top_t_idx]
                    corrective_steps.append(top_t_name)
                
                print(f" {t_name}  NON-CONFORMANT at marking {current_marking_idx}")
                print(f" -> Corrective path: {corrective_steps}")
                print(f" -> Then fire {t_name}, land at marking {best_tgt_idx}")
                
                prefix_alignment.append((t_name, 'NON-CONFORMANT', corrective_steps))
                current_marking_idx = best_tgt_idx  
            
    return prefix_alignment

event_stream = ['t1', 't5', 't2', 't7']  # t5 is non-conformant after t1
print("=== Online Conformance Checking ===")
result = online_conformance_check(
    event_stream, model, reachability_graph,
    transition_to_enabled_markings, marking_to_idx
)
print(result)