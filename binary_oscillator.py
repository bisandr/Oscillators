import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Train the binary oscillator forecaster.")
    parser.add_argument("data_path", nargs="?", default="data.csv", help="CSV file to load.")
    parser.add_argument(
        "--picks",
        type=int,
        required=True,
        help="Number of active entries expected in each row and predicted for the next step.",
    )
    return parser.parse_args()


def load_Y(path, active_count, device="cpu"):
    df = pd.read_csv(path, header=None)
    X = df.to_numpy(dtype=np.float32)
    if X.shape[1] < active_count:
        raise ValueError(f"CSV has {X.shape[1]} columns but --picks={active_count}.")
    Y = (X > 0.5).astype(np.float32)

    row_sums = Y.sum(axis=1)
    bad = np.where(row_sums != active_count)[0]
    if bad.size > 0:
        i = int(bad[0])
        raise ValueError(f"Row {i} sums to {row_sums[i]} (expected {active_count}).")

    Yt = torch.tensor(Y, device=device)
    if Yt.shape[0] < 2:
        raise ValueError(f"Need at least 2 rows; got T={Yt.shape[0]}.")
    return Yt

def ranking_loss(scores, y_true, neg_per_pos=20):
    pos_idx = torch.where(y_true > 0.5)[0]
    neg_candidates = torch.where(y_true < 0.5)[0]
    Q = neg_per_pos * pos_idx.numel()
    neg_idx = neg_candidates[torch.randint(0, neg_candidates.numel(), (Q,), device=scores.device)]

    pos_scores = scores[pos_idx]
    neg_scores = scores[neg_idx]
    diff = pos_scores[:, None] - neg_scores[None, :]
    return F.softplus(-diff).mean()

class TopKGRU(nn.Module):
    def __init__(self, N, hidden=32):
        super().__init__()
        self.gru = nn.GRU(input_size=N, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, N)
        self.carryover = nn.Parameter(torch.tensor(0.0))

    def scores_next(self, h_prev, y_prev):
        return self.out(h_prev) + self.carryover * y_prev

def train_from_scratch(model, Y, epochs=120, lr=1e-3, neg_per_pos=20, weight_decay=1e-4):
    T, N = Y.shape
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Y_batch = Y.unsqueeze(0)  # [1, T, N]

    for ep in range(epochs):
        opt.zero_grad()
        h_seq, _ = model.gru(Y_batch)  # [1, T, hidden]

        loss = 0.0
        steps = 0
        for t in range(1, T):
            h_prev = h_seq[0, t-1]
            scores = model.scores_next(h_prev, Y[t-1])
            loss = loss + ranking_loss(scores, Y[t], neg_per_pos=neg_per_pos)
            steps += 1

        loss = loss / steps
        loss.backward()
        opt.step()

        if (ep + 1) % 20 == 0:
            carryover = model.carryover.item()
            print(f"epoch={ep+1:3d}/{epochs} loss={loss.item():.4f} carryover={carryover:.3f}")

    return model

@torch.no_grad()
def forecast_one_step(model, Y, active_count):
    Y_batch = Y.unsqueeze(0)
    h_seq, _ = model.gru(Y_batch)
    h_last = h_seq[0, -1]
    scores = model.scores_next(h_last, Y[-1])
    topk = torch.topk(scores, k=active_count).indices
    return topk, scores

def main():
    args = parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Y = load_Y(args.data_path, args.picks, device=device)
    Y = Y.float()
    T, N = Y.shape
    print(f"Loaded {args.data_path} with T={T}, N={N}")

    # ── Ensemble over multiple seeds ──────────────────────────────────────
    N_ENSEMBLE = 5
    all_scores = []

    for seed in range(N_ENSEMBLE):
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = TopKGRU(N=N, hidden=32).to(device)
        model = train_from_scratch(model, Y, epochs=120, lr=1e-3)

        _, scores = forecast_one_step(model, Y, args.picks)
        all_scores.append(scores)

        topk_seed = torch.topk(scores, k=args.picks).indices
        print(f"[Seed {seed}] picks (1-based): {[i+1 for i in topk_seed.tolist()]}")

    # ── Average scores across all models ─────────────────────────────────
    avg_scores = torch.stack(all_scores).mean(dim=0)   # [N]
    topk = torch.topk(avg_scores, k=args.picks).indices

    print(f"\nEnsemble Forecast top-{args.picks} (0-based): {topk.tolist()}")
    print(f"Ensemble Forecast top-{args.picks} (1-based): {[i+1 for i in topk.tolist()]}")

    # ── Confidence report — how many seeds voted for each top-10 component ─
    TOP_N_REPORT = 10
    sorted_idx = torch.argsort(avg_scores, descending=True)
    seed_topk_sets = [set(torch.topk(s, k=args.picks).indices.tolist()) for s in all_scores]
    print(f"\nConfidence report (top {TOP_N_REPORT} components):")
    for rank, idx in enumerate(sorted_idx[:TOP_N_REPORT]):
        i = idx.item()
        votes = sum(1 for topk_set in seed_topk_sets if i in topk_set)
        print(f"  #{rank+1:2d}  component {i+1:3d}  avg_score={avg_scores[i]:.4f}  votes={votes}/{N_ENSEMBLE}")

if __name__ == "__main__":
    main()