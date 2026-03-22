import torch.optim as optim
import torch
import torch.nn.functional as F
from model import PetriNetAlignmentPredictor
from reachability_graph import *
from dataset import *

model = PetriNetAlignmentPredictor(reachability_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def training_step(model, X_src_train, X_tgt_train, y_alphas_train, epoch):
    model.train()
    epoch_loss = 0

    tau = max(0.1, 1.0 - epoch * 0.005)  

    for i in range(len(X_src_train)):
        optimizer.zero_grad()

        v_s = X_src_train[i:i+1]
        v_t = X_tgt_train[i:i+1]
        y_true = y_alphas_train[i]

        if y_true.dim() == 3:
            y_true = y_true.squeeze(1)   # → [L, num_t]

        target_indices = y_true.argmax(dim=-1).long()  # [L]

        v_pred, pred_seq = model(v_s, v_t, training=True, tau=tau)

        min_len = min(pred_seq.size(0), target_indices.size(0))

        if min_len > 0:
            loss_ce  = F.cross_entropy(pred_seq[:min_len], target_indices[:min_len])
            loss_pos = F.mse_loss(v_pred, v_t)
            loss = loss_ce + 0.1 * loss_pos
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(X_src_train)

epochs = 200
for epoch in range(epochs):
    avg_loss = training_step(model, X_src_train, X_tgt_train, y_alphas_train, epoch)  
    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

# Évaluation
model.eval()
with torch.no_grad():
    for i in range(len(X_src_test)):
        v_src_test = X_src_test[i:i+1]
        v_tgt_test = X_tgt_test[i:i+1]

        v_pred, pred_logits = model(v_src_test, v_tgt_test, training=False)

        src_idx  = v_src_test.argmax().item()
        tgt_idx  = v_tgt_test.argmax().item()
        pred_idx = v_pred.argmax().item()

        print(f"\n--- Test Chemin {i} ---")
        print(f"Source (Index) : {src_idx}")
        print(f"Cible  (Index) : {tgt_idx}")
        print(f"Prédit (Index) : {pred_idx}  {'SUCCESS' if pred_idx == tgt_idx else 'FAILED'}")

        for step in range(pred_logits.size(0)):
            probs = F.softmax(pred_logits[step], dim=-1)
            top_t = probs.argmax().item()
            conf  = probs[top_t].item()
            if conf > 0.1:
                print(f"Pas {step+1}: Transition t{top_t+1} (confiance={conf:.2f})")