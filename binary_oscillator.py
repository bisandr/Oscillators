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


def compute_last_seen(Y):
    """Return the last timestep index at which each component was active (-1 if never)."""
    T, N = Y.shape
    last_seen = torch.full((N,), -1.0, device=Y.device)
    for t in range(T):
        active = (Y[t] > 0.5).nonzero(as_tuple=True)[0]
        last_seen[active] = float(t)
    return last_seen


def compute_gap_features(Y):
    T, N = Y.shape
    gaps = torch.zeros_like(Y)
    last_seen = torch.full((N,), -1.0)
    for t in range(T):
        for n in range(N):
            if Y[t, n] > 0.5:
                last_seen[n] = float(t)
            gaps[t, n] = (t - last_seen[n]) if last_seen[n] >= 0 else float(t)
    return gaps


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
    freq = Yt.mean(dim=0, keepdim=True).expand_as(Yt)
    gaps = compute_gap_features(Yt).to(device)
    Y_input = torch.cat([Yt, freq, gaps], dim=1)  # [T, 3N]
    return Yt, Y_input


def ranking_loss(scores, y_true, neg_per_pos=50, margin=1.0):
    pos_idx = torch.where(y_true > 0.5)[0]
    neg_candidates = torch.where(y_true < 0.5)[0]
    Q = neg_per_pos * pos_idx.numel()
    neg_idx = neg_candidates[
        torch.randint(0, neg_candidates.numel(), (Q,), device=scores.device)
    ]
    pos_scores = scores[pos_idx]
    neg_scores = scores[neg_idx]
    diff = pos_scores[:, None] - neg_scores[None, :]
    return F.softplus(margin - diff).mean()


class TopKGRU(nn.Module):
    def __init__(self, N, input_size=None, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.N = N
        in_sz = input_size if input_size is not None else N
        self.gru = nn.GRU(
            input_size=in_sz,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # explicit inter-component interaction (N×N learned coupling matrix)
        self.interaction = nn.Linear(N, N, bias=False)
        nn.init.zeros_(self.interaction.weight)  # start neutral; learned from data
        # MLP scoring head
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, N),
        )
        self.carryover = nn.Parameter(torch.tensor(0.0))

    def scores_next(self, h_prev, y_prev):
        return self.out(h_prev) + self.interaction(y_prev) + self.carryover * y_prev


def train_from_scratch(model, Y_input, Y_target, epochs=200, lr=1e-3,
                       neg_per_pos=50, weight_decay=1e-4, patience=20):
    T = Y_target.shape[0]
    split = int(T * 0.85)
    Y_in_train,  Y_in_val  = Y_input[:split],  Y_input[split:]
    Y_tgt_train, Y_tgt_val = Y_target[:split], Y_target[split:]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, no_improve, best_state = float("inf"), 0, None

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        h_seq, _ = model.gru(Y_in_train.unsqueeze(0))
        loss = sum(
            ranking_loss(model.scores_next(h_seq[0, t-1], Y_tgt_train[t-1]),
                         Y_tgt_train[t], neg_per_pos=neg_per_pos)
            for t in range(1, split)
        ) / (split - 1)
        loss.backward()
        opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            h_val, _ = model.gru(Y_in_val.unsqueeze(0))
            T_val = Y_tgt_val.shape[0]
            val_loss = sum(
                ranking_loss(model.scores_next(h_val[0, t-1], Y_tgt_val[t-1]),
                             Y_tgt_val[t])
                for t in range(1, T_val)
            ) / (T_val - 1)
            val_loss = val_loss.item()

        if val_loss < best_val - 1e-5:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {ep+1} (best val={best_val:.4f})")
                break

        if (ep + 1) % 20 == 0:
            print(f"epoch={ep+1:3d}/{epochs} train={loss.item():.4f} "
                  f"val={val_loss:.4f} carryover={model.carryover.item():.3f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def forecast_one_step(model, Y_input, Y_target, active_count, mc_samples=50):
    all_probs = []
    was_training = model.training
    model.train()  # keep dropout ON for MC sampling
    for _ in range(mc_samples):
        h_seq, _ = model.gru(Y_input.unsqueeze(0))
        scores = model.scores_next(h_seq[0, -1], Y_target[-1])
        all_probs.append(torch.sigmoid(scores))
    model.train(was_training)  # restore original mode
    prob_stack = torch.stack(all_probs)          # [mc_samples, N]
    mean_probs = prob_stack.mean(dim=0)          # [N]
    std_probs  = prob_stack.std(dim=0)           # [N]
    topk = torch.topk(mean_probs, k=active_count).indices
    return topk, mean_probs, std_probs


@torch.no_grad()
def simulate_future(model, Y_input, Y_target, active_count, steps=10):
    model.eval()
    T_hist = Y_target.shape[0]
    N = Y_target.shape[1]
    simulated = []
    y_cur = Y_target[-1].clone()
    _, h_state = model.gru(Y_input.unsqueeze(0))
    # h_state shape: [num_layers, 1, hidden] — keep as-is for nn.GRU

    # Frequency is fixed (historical mean); gaps evolve step-by-step
    freq = Y_target.mean(dim=0)  # [N]
    last_seen = compute_last_seen(Y_target)      # [N], reuse helper
    cur_t = float(T_hist - 1)

    for _ in range(steps):
        cur_t += 1.0
        # Use only the last layer hidden state for scoring
        h_top = h_state[-1, 0]   # [hidden]
        scores = model.scores_next(h_top, y_cur)
        probs  = torch.sigmoid(scores)
        sampled = torch.zeros_like(probs)
        sampled[torch.multinomial(probs, num_samples=active_count, replacement=False)] = 1.0
        simulated.append(sampled.clone())

        # Update last_seen for active components (vectorized)
        active_mask = sampled > 0.5
        last_seen = torch.where(active_mask, torch.tensor(cur_t, device=Y_target.device), last_seen)

        # Build full input: [binary, freq, gaps] (vectorized)
        gaps = torch.where(last_seen >= 0, cur_t - last_seen,
                           torch.tensor(cur_t, device=Y_target.device))
        x_next = torch.cat([sampled, freq, gaps], dim=0)  # [3N]
        _, h_state = model.gru(x_next.unsqueeze(0).unsqueeze(0), h_state)
        y_cur = sampled

    return torch.stack(simulated)  # [steps, N]


def plot_latent_trajectory(model, Y_input):
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("matplotlib / sklearn not available — skipping latent trajectory plot.")
        return

    model.eval()
    with torch.no_grad():
        h_seq, _ = model.gru(Y_input.unsqueeze(0))
    H = h_seq[0].cpu().numpy()          # [T, hidden]

    pca = PCA(n_components=2)
    H2d = pca.fit_transform(H)          # [T, 2]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(H2d[:, 0], label="PC1")
    plt.plot(H2d[:, 1], label="PC2")
    plt.title("Latent State Trajectory (PCA)")
    plt.xlabel("Time step")
    plt.legend()

    plt.subplot(1, 2, 2)
    sc = plt.scatter(H2d[:, 0], H2d[:, 1],
                     c=range(len(H2d)), cmap="viridis", s=10)
    plt.colorbar(sc, label="Time")
    plt.title("Phase Space")
    plt.tight_layout()
    plt.savefig("latent_trajectory.png", dpi=150)
    print("Saved latent_trajectory.png")


def main():
    args = parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Y_target, Y_input = load_Y(args.data_path, args.picks, device=device)
    Y_target = Y_target.float()
    Y_input  = Y_input.float()
    T, N = Y_target.shape
    input_size = Y_input.shape[1]   # 3N
    print(f"Loaded {args.data_path}: T={T}, N={N}, input_size={input_size}")

    # ── Ensemble: train 5 models with different seeds ─────────────────────
    N_ENSEMBLE = 5
    all_mean_probs = []
    all_std_probs  = []
    last_model     = None

    for seed in range(N_ENSEMBLE):
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = TopKGRU(N=N, input_size=input_size, hidden=64).to(device)
        model = train_from_scratch(model, Y_input, Y_target, epochs=200, lr=1e-3)

        topk_seed, mean_probs, std_probs = forecast_one_step(
            model, Y_input, Y_target, args.picks
        )
        all_mean_probs.append(mean_probs)
        all_std_probs.append(std_probs)
        last_model = model

        print(f"[Seed {seed}] picks (1-based): {[i+1 for i in topk_seed.tolist()]}")

    # ── Aggregate ensemble ────────────────────────────────────────────────
    avg_probs = torch.stack(all_mean_probs).mean(dim=0)   # [N]
    avg_std   = torch.stack(all_std_probs).mean(dim=0)    # [N]
    topk      = torch.topk(avg_probs, k=args.picks).indices

    print(f"\nEnsemble Forecast top-{args.picks} (0-based): {topk.tolist()}")
    print(f"Ensemble Forecast top-{args.picks} (1-based): {[i+1 for i in topk.tolist()]}")

    # ── Confidence report ─────────────────────────────────────────────────
    sorted_idx = torch.argsort(avg_probs, descending=True)
    topk_sets = [set(torch.topk(mp, k=args.picks).indices.tolist()) for mp in all_mean_probs]
    print(f"\nConfidence report (top 10 components):")
    for rank, idx in enumerate(sorted_idx[:10]):
        i = idx.item()
        votes = sum(1 for topk_set in topk_sets if i in topk_set)
        print(f"  #{rank+1:2d}  component {i+1:3d}  "
              f"P={avg_probs[i]:.3f} ± {avg_std[i]:.3f}  votes={votes}/{N_ENSEMBLE}")

    # ── Latent trajectory ─────────────────────────────────────────────────
    plot_latent_trajectory(last_model, Y_input)

    # ── Simulation: next 5 steps ──────────────────────────────────────────
    sim = simulate_future(last_model, Y_input, Y_target, args.picks, steps=5)
    print(f"\nSimulated next 5 steps (1-based active components):")
    for t, row in enumerate(sim):
        active = (row > 0.5).nonzero(as_tuple=True)[0].tolist()
        print(f"  t+{t+1}: {[i+1 for i in active]}")

if __name__ == "__main__":
    main()