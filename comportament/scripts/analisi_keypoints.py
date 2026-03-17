###############################################################
# CHICKEN KEYPOINT TRACKING AND BEHAVIOUR ANALYSIS
# -------------------------------------------------------------
# This script loads chicken pose keypoint data (from the CSV
# produced by SimulaMovimiento1pollo_ClassificaYCuantificaComportamientos.R
# or from a real tracker output), extracts movement features,
# classifies behaviours using a Random Forest, and produces
# visualisations of keypoint trajectories and time budgets.
###############################################################

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── reproducibility ────────────────────────────────────────────
SEED = 123
np.random.seed(SEED)

###############################################################
# 1. KEYPOINT NAMES AND BEHAVIOUR LABELS
###############################################################

KEYPOINTS = ["tail", "neck", "head"]  # 3 tracked body parts
BEHAVIOURS = ["inactive", "locomotion", "foraging", "pecking", "preening"]

###############################################################
# 2. FEATURE EXTRACTION FROM KEYPOINT TRAJECTORIES
###############################################################

def extract_features(df: pd.DataFrame, fps: int = 10) -> pd.DataFrame:
    """
    Compute movement features for each frame from raw keypoint positions.

    Parameters
    ----------
    df : DataFrame with columns neck_x, neck_y, head_x, head_y, tail_x, tail_y
    fps : frames per second (used to set rolling-window sizes in seconds)

    Returns
    -------
    DataFrame with one feature column per derived metric.
    """
    feat = pd.DataFrame(index=df.index)

    # ── body and head vectors ───────────────────────────────────
    body_dx = df["neck_x"] - df["tail_x"]  # tail → neck
    body_dy = df["neck_y"] - df["tail_y"]
    head_dx = df["head_x"] - df["neck_x"]  # neck → head
    head_dy = df["head_y"] - df["neck_y"]

    body_len = np.sqrt(body_dx**2 + body_dy**2).replace(0, np.nan)

    # ── frame-level velocities ──────────────────────────────────
    neck_vx = df["neck_x"].diff().fillna(0)
    neck_vy = df["neck_y"].diff().fillna(0)
    feat["vel_body"] = np.sqrt(neck_vx**2 + neck_vy**2)

    head_vx = df["head_x"].diff().fillna(0)
    head_vy = df["head_y"].diff().fillna(0)
    feat["vel_head"] = np.sqrt(head_vx**2 + head_vy**2)

    # ── head acceleration ───────────────────────────────────────
    feat["acc_head"] = feat["vel_head"].diff().fillna(0)

    # ── head vector length ──────────────────────────────────────
    feat["head_length"] = np.sqrt(head_dx**2 + head_dy**2)

    # ── parallel / perpendicular head components ────────────────
    # How much the head moves along vs across the body axis
    feat["parallel"] = (
        head_dx * body_dx + head_dy * body_dy
    ) / body_len
    feat["perpendicular"] = (
        head_dx * (-body_dy) + head_dy * body_dx
    ) / body_len

    # ── rolling-window movement sums ────────────────────────────
    win_2s = max(1, int(fps * 2))   # ~2 seconds
    win_5s = max(1, int(fps * 5))   # ~5 seconds
    feat["dist_body_20"] = (
        feat["vel_body"].rolling(win_2s, min_periods=1).sum()
    )
    feat["dist_body_50"] = (
        feat["vel_body"].rolling(win_5s, min_periods=1).sum()
    )

    # ── head-body angle variance ────────────────────────────────
    head_angle = np.arctan2(head_dy, head_dx)
    body_angle = np.arctan2(body_dy, body_dx)
    rel_angle = head_angle - body_angle
    feat["angle_var"] = (
        rel_angle.rolling(win_2s, min_periods=1).var().fillna(0)
    )

    # ── head vs body speed ratio ────────────────────────────────
    feat["ratio_head_body"] = feat["vel_head"] / (feat["vel_body"] + 1e-6)

    return feat.fillna(0)


###############################################################
# 3. SIMULATION OF KEYPOINT DATA (fallback when no CSV is found)
###############################################################

def simulate_keypoints(n_frames: int = 3600, fps: int = 10) -> pd.DataFrame:
    """
    Generate a synthetic single-chicken keypoint dataset using a
    Markov-chain behaviour model.  Used as a fallback when no real
    or pre-computed CSV is available.

    Parameters
    ----------
    n_frames : number of frames to simulate
    fps      : frames per second

    Returns
    -------
    DataFrame with columns: frame, state, tail_x, tail_y,
                             neck_x, neck_y, head_x, head_y
    """
    rng = np.random.default_rng(SEED)

    # ── Markov transition matrix ────────────────────────────────
    trans = np.array([
        [0.60, 0.20, 0.10, 0.05, 0.05],  # inactive
        [0.20, 0.20, 0.40, 0.10, 0.10],  # locomotion
        [0.30, 0.10, 0.40, 0.10, 0.10],  # foraging
        [0.20, 0.20, 0.20, 0.30, 0.10],  # pecking
        [0.30, 0.10, 0.20, 0.10, 0.30],  # preening
    ])
    min_dur = {"inactive": 30, "locomotion": 20,
               "foraging": 25, "pecking": 5, "preening": 15}

    # ── generate behaviour sequence ─────────────────────────────
    states_seq = []
    cur = rng.integers(len(BEHAVIOURS))
    while len(states_seq) < n_frames:
        label = BEHAVIOURS[cur]
        dur = int(min_dur[label] + rng.exponential(min_dur[label] * 2))
        states_seq.extend([label] * dur)
        cur = rng.choice(len(BEHAVIOURS), p=trans[cur])
    states_seq = states_seq[:n_frames]

    # ── simulate neck (body-centre) trajectory ──────────────────
    step = np.where(
        np.array(states_seq) == "locomotion",
        rng.normal(0, 0.05, n_frames),
        rng.normal(0, 0.01, n_frames),
    )
    neck_x = np.cumsum(step)
    neck_y = np.cumsum(
        np.where(
            np.array(states_seq) == "locomotion",
            rng.normal(0, 0.05, n_frames),
            rng.normal(0, 0.01, n_frames),
        )
    )
    # light smoothing
    neck_x = uniform_filter1d(neck_x, size=5)
    neck_y = uniform_filter1d(neck_y, size=5)

    # ── tail (fixed offset behind neck) ─────────────────────────
    tail_x = neck_x - 0.3 + rng.normal(0, 0.005, n_frames)
    tail_y = neck_y + rng.normal(0, 0.005, n_frames)

    # ── head (behaviour-dependent offsets) ──────────────────────
    head_x = neck_x + 0.15 + rng.normal(0, 0.01, n_frames)
    head_y = neck_y + 0.02 + rng.normal(0, 0.01, n_frames)

    peck_mask = np.array(states_seq) == "pecking"
    head_y[peck_mask] = neck_y[peck_mask] - np.abs(
        rng.normal(0.05, 0.01, peck_mask.sum())
    )
    preen_mask = np.array(states_seq) == "preening"
    head_x[preen_mask] = neck_x[preen_mask] - np.abs(
        rng.normal(0.10, 0.01, preen_mask.sum())
    )
    head_x = uniform_filter1d(head_x, size=5)
    head_y = uniform_filter1d(head_y, size=5)

    return pd.DataFrame(
        {
            "frame": np.arange(1, n_frames + 1),
            "state": states_seq,
            "tail_x": tail_x,
            "tail_y": tail_y,
            "neck_x": neck_x,
            "neck_y": neck_y,
            "head_x": head_x,
            "head_y": head_y,
        }
    )


###############################################################
# 4. BEHAVIOUR CLASSIFIER
###############################################################

class ChickenKeypointAnalyser:
    """
    Encapsulates the full pipeline: feature extraction → training
    → prediction → visualisation for a single-chicken keypoint
    time-series.
    """

    def __init__(self, fps: int = 10, n_trees: int = 300):
        self.fps = fps
        self.n_trees = n_trees
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=n_trees, random_state=SEED, n_jobs=-1
        )
        self._feature_cols: list[str] = []

    # ── fit ─────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "ChickenKeypointAnalyser":
        """
        Fit the classifier on a labelled keypoint DataFrame.

        Parameters
        ----------
        df : DataFrame containing keypoint columns plus a 'state' column
             with ground-truth behaviour labels.
        """
        features = extract_features(df, fps=self.fps)
        self._feature_cols = features.columns.tolist()

        X = self.scaler.fit_transform(features.values)
        y = df["state"].values
        self.clf.fit(X, y)
        return self

    # ── predict ─────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict behaviour labels for every frame in *df*.

        Parameters
        ----------
        df : DataFrame containing keypoint columns (no 'state' required).

        Returns
        -------
        Array of predicted behaviour strings.
        """
        features = extract_features(df, fps=self.fps)[self._feature_cols]
        X = self.scaler.transform(features.values)
        return self.clf.predict(X)

    # ── evaluate ────────────────────────────────────────────────
    def evaluate(
        self, df: pd.DataFrame, output_path: str = "."
    ) -> None:
        """
        Print classification report and save a confusion-matrix PNG.
        """
        y_true = df["state"].values
        y_pred = self.predict(df)

        print("\n=== Classification Report ===")
        print(
            classification_report(
                y_true, y_pred, labels=BEHAVIOURS, zero_division=0
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=BEHAVIOURS)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=BEHAVIOURS)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion matrix – behaviour classification")
        plt.tight_layout()
        out = os.path.join(output_path, "confusion_matrix.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved → {out}")

    # ── time budget ─────────────────────────────────────────────
    def time_budget(
        self, predictions: np.ndarray, output_path: str = "."
    ) -> pd.Series:
        """
        Compute and plot the estimated time budget (minutes per behaviour).

        Parameters
        ----------
        predictions : array of predicted behaviour strings (one per frame)
        output_path : directory where the PNG is saved

        Returns
        -------
        pandas Series with minutes spent per behaviour.
        """
        counts = pd.Series(predictions).value_counts().reindex(BEHAVIOURS, fill_value=0)
        minutes = counts / self.fps / 60

        colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(minutes.index, minutes.values, color=colours)
        ax.set_ylabel("Minutes")
        ax.set_title("Estimated behavioural time budget")
        plt.tight_layout()
        out = os.path.join(output_path, "behaviour_time_budget.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Time-budget plot saved → {out}")
        return minutes

    # ── trajectory visualisation ─────────────────────────────────
    def plot_trajectories(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        output_path: str = ".",
        max_frames: int = 3000,
    ) -> None:
        """
        Plot keypoint trajectories colour-coded by predicted behaviour.

        Parameters
        ----------
        df           : raw keypoint DataFrame
        predictions  : predicted behaviour label for each frame
        output_path  : directory where the PNG is saved
        max_frames   : number of frames to plot (default 3 000; at 10 fps
                       this corresponds to approximately 5 minutes)
        """
        n = min(len(df), max_frames)
        df_plot = df.iloc[:n].copy()
        pred_plot = predictions[:n]

        colour_map = {b: c for b, c in zip(
            BEHAVIOURS,
            ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"],
        )}

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_aspect("equal")
        ax.set_title(
            f"Keypoint trajectories (first {n} frames, colour = predicted behaviour)"
        )

        # draw neck (body-centre) trajectory coloured by behaviour
        for i in range(n - 1):
            colour = colour_map.get(pred_plot[i], "grey")
            ax.plot(
                df_plot["neck_x"].iloc[i : i + 2],
                df_plot["neck_y"].iloc[i : i + 2],
                color=colour,
                linewidth=0.8,
                alpha=0.7,
            )

        # scatter keypoints at the final frame
        last = df_plot.iloc[-1]
        ax.scatter(last["tail_x"], last["tail_y"], s=60, marker="s",
                   color="k", zorder=5, label="tail (last frame)")
        ax.scatter(last["neck_x"], last["neck_y"], s=60, marker="o",
                   color="k", zorder=5, label="neck (last frame)")
        ax.scatter(last["head_x"], last["head_y"], s=60, marker="^",
                   color="k", zorder=5, label="head (last frame)")

        # legend – behaviour colours
        handles = [
            plt.Line2D([0], [0], color=c, linewidth=2, label=b)
            for b, c in colour_map.items()
        ]
        handles += ax.get_legend_handles_labels()[0]
        ax.legend(handles=handles, loc="upper left", fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        out = os.path.join(output_path, "keypoint_trajectories.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Trajectory plot saved → {out}")

    # ── keypoint skeleton visualisation ─────────────────────────
    def plot_keypoint_snapshots(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        output_path: str = ".",
        n_snapshots: int = 10,
        stride: int | None = None,
    ) -> None:
        """
        Draw a grid of keypoint skeleton snapshots spaced evenly
        throughout the recording.

        Each panel shows the three keypoints (tail, neck, head)
        connected as a skeleton and labelled with the predicted
        behaviour.

        Parameters
        ----------
        df           : raw keypoint DataFrame
        predictions  : predicted behaviour label for each frame
        output_path  : directory where the PNG is saved
        n_snapshots  : number of panels in the grid
        stride       : frame stride; if None, computed automatically
        """
        n = len(df)
        if stride is None:
            stride = max(1, n // n_snapshots)
        indices = list(range(0, n, stride))[:n_snapshots]

        cols = 5
        rows = (len(indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).flatten()

        colour_map = {b: c for b, c in zip(
            BEHAVIOURS,
            ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"],
        )}

        for ax, idx in zip(axes, indices):
            row = df.iloc[idx]
            pred = predictions[idx]
            colour = colour_map.get(pred, "grey")

            xs = [row["tail_x"], row["neck_x"], row["head_x"]]
            ys = [row["tail_y"], row["neck_y"], row["head_y"]]

            ax.plot(xs, ys, "-o", color=colour, linewidth=2, markersize=6)
            ax.annotate("T", (row["tail_x"], row["tail_y"]),
                        fontsize=7, ha="center", va="bottom")
            ax.annotate("N", (row["neck_x"], row["neck_y"]),
                        fontsize=7, ha="center", va="bottom")
            ax.annotate("H", (row["head_x"], row["head_y"]),
                        fontsize=7, ha="center", va="bottom")
            ax.set_title(f"frame {idx}\n{pred}", fontsize=8, color=colour)
            ax.set_aspect("equal")
            ax.axis("off")

        # hide unused panels
        for ax in axes[len(indices):]:
            ax.axis("off")

        fig.suptitle("Keypoint skeleton snapshots", fontsize=12)
        plt.tight_layout()
        out = os.path.join(output_path, "keypoint_snapshots.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Snapshot grid saved → {out}")


###############################################################
# 5. MAIN ENTRY POINT
###############################################################

def main() -> None:
    # ── locate data ─────────────────────────────────────────────
    csv_path = os.environ.get(
        "CHICKEN_CSV",
        os.path.join(os.path.dirname(__file__), "chicken_simulated_data_full.csv"),
    )
    output_path = os.environ.get(
        "CHICKEN_OUTPUT",
        os.path.dirname(__file__),
    )

    if os.path.exists(csv_path):
        print(f"Loading keypoint data from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("No CSV found – generating simulated keypoint data …")
        n_frames = int(os.environ.get("CHICKEN_FRAMES", 3600))
        fps = int(os.environ.get("CHICKEN_FPS", 10))
        df = simulate_keypoints(n_frames=n_frames, fps=fps)
        sim_out = os.path.join(output_path, "chicken_simulated_data_full.csv")
        df.to_csv(sim_out, index=False)
        print(f"Simulated data saved → {sim_out}")

    fps = int(os.environ.get("CHICKEN_FPS", 10))

    # ── fit and predict ──────────────────────────────────────────
    print("\nExtracting features and training classifier …")
    analyser = ChickenKeypointAnalyser(fps=fps)

    if "state" in df.columns:
        analyser.fit(df)
        predictions = analyser.predict(df)
        print("\nEvaluating against ground-truth labels …")
        analyser.evaluate(df, output_path=output_path)
    else:
        # No ground-truth: use predictions as labels for fitting
        # (unsupervised mode – cluster-based labels would be needed
        #  in a real deployment; here we fit on raw position heuristics)
        raise ValueError(
            "Input CSV must include a 'state' column with ground-truth "
            "behaviour labels for supervised training.  "
            "Run the R simulation script first to generate labelled data."
        )

    # ── time budget ──────────────────────────────────────────────
    budget = analyser.time_budget(predictions, output_path=output_path)
    print("\n=== Time budget (minutes) ===")
    print(budget.to_string())

    # ── visualisations ───────────────────────────────────────────
    print("\nGenerating visualisations …")
    analyser.plot_trajectories(df, predictions, output_path=output_path)
    analyser.plot_keypoint_snapshots(df, predictions, output_path=output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
