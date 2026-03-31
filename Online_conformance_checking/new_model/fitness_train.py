import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import json
from fitness_model import LSTMFitnessModel

def group_by_case(dataset):
    cases = defaultdict(list)
    for sample in dataset:
        cases[sample['case_id']].append(sample)
    return cases

def train(model, dataset, act_to_idx, num_epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    cases = group_by_case(dataset)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for case_id, samples in cases.items():
            h, c = None, None
            optimizer.zero_grad()
            case_loss = torch.tensor(0.0)

            for sample in samples:
                new_activity = sample['prefix'][-1]
                fitness = sample['fitness']

                idx = act_to_idx.get(new_activity, 0)
                x = torch.tensor([[idx]], dtype=torch.long)
                length = torch.tensor([1], dtype=torch.long)
                y = torch.tensor([[fitness]], dtype=torch.float)

                pred, (h, c) = model(x, length, h, c)
                loss = criterion(pred, y)
                case_loss = case_loss + loss

                h = h.detach()
                c = c.detach()

            case_loss.backward()
            optimizer.step()
            total_loss += case_loss.item()

        avg_loss = total_loss / len(cases)
        print(f"Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}")

def evaluate(model, dataset, act_to_idx, criterion):
    model.eval()
    cases = group_by_case(dataset)
    total_loss = 0.0
    
    with torch.no_grad():
        for case_id, samples in cases.items():
            h, c = None, None
            for sample in samples:
                new_activity = sample['prefix'][-1]
                fitness = sample['fitness']
                idx = act_to_idx.get(new_activity, 0)
                x = torch.tensor([[idx]], dtype=torch.long)
                length = torch.tensor([1], dtype=torch.long)
                y = torch.tensor([[fitness]], dtype=torch.float)
                pred, (h, c) = model(x, length, h, c)
                loss = criterion(pred, y)
                total_loss += loss.item()
    
    avg_loss = total_loss / sum(len(s) for s in cases.values())
    print(f"Test MSE: {avg_loss:.4f}")
    model.train()
    return avg_loss

if __name__ == "__main__":
    dataset_path = r"C:\Users\LENONVO\OneDrive\Desktop\graphs\sujet-CRAN\datasets\spesis\ground_truth_dataset.json"
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    all_activities = list({a for s in dataset for a in s['prefix']})
    act_to_idx = {act: i+1 for i, act in enumerate(all_activities)}
    vocab_size = len(act_to_idx) + 1

    import random

    # split by case not by sample
    all_case_ids = list(group_by_case(dataset).keys())
    random.shuffle(all_case_ids)

    split = int(0.8 * len(all_case_ids))
    train_case_ids = set(all_case_ids[:split])
    test_case_ids = set(all_case_ids[split:])

    train_dataset = [s for s in dataset if s['case_id'] in train_case_ids]
    test_dataset = [s for s in dataset if s['case_id'] in test_case_ids]

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    model = LSTMFitnessModel(vocab_size=vocab_size)
    train(model, train_dataset, act_to_idx)
    torch.save(model.state_dict(), "lstm_fitness_baseline.pt")
    print("Model saved.")


    criterion = nn.MSELoss()
    evaluate(model, test_dataset, act_to_idx, criterion)