"""
Frozen-feature stop probes on the audit-cleaned rows (read only; results in this folder).

Question: the stop outputs trained so far stop early at mid-track pictures that show no line ahead. Do "hard negatives"
help, that is, giving extra loss weight to non-stop pictures that look like an end? Every variant trains a stop-only
head (one hidden layer of 256, as training_probes.ProbeHead) on frozen ResNet-18 grid features
(03_data/processed/v01_training_index/features/resnet18_imagenet_grid.npy), with leave-one-sheet-out folds, early
stopping on 15 % of each training sheet's episodes, and the same per-episode stop scoring as training_metrics:
    A     positives: the 259 stop pictures (positive weight 10), as in the final candidate;
    B5    A, plus weight 5 on non-stop pictures more than 3 cm from the goal that show no line in their top half
          (the audit's line_share_top_half of at most 0.002; a training weight only, nothing at run time);
    B10   the same with weight 10;
    C     positives: every picture within 2 cm of the goal (positive weight 3);
    D     C plus the B5 weights;
    E     mined hard negatives: train A, then give weight 5 to the training pictures more than 3 cm from the goal on
          which A's stop probability is at least 0.2, and train again (no picture rule needed).
Writes stop_probe.out.txt and stop_probe_summary.csv.
"""

import json
import math
import sys
import time

import numpy
import pandas
import torch

from common_frames import AUDIT_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER, audit_cleaned, read_frames
from stop_analysis import episode_outcomes, shares

THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
lines = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text, flush = True)


class StopHead(torch.nn.Module):

    def __init__(self, mean: numpy.ndarray, spread: numpy.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype = torch.float32))
        self.register_buffer("spread", torch.as_tensor(spread, dtype = torch.float32))
        self.layers = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(len(mean), 256), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers((features.float() - self.mean) / self.spread)[:, 0]


def train_head(features, target, weight, positive_weight, train_rows, validation_rows, seed):
    torch.manual_seed(seed)
    generator = numpy.random.default_rng(seed)
    sample = features[train_rows[::max(1, len(train_rows) // 4000)]].astype(numpy.float64)
    head = StopHead(sample.mean(axis = 0), sample.std(axis = 0) + 1e-3)
    optimiser = torch.optim.AdamW(head.parameters(), lr = 1e-3, weight_decay = 1e-2)
    feature_tensor = torch.from_numpy(features)
    target_tensor = torch.from_numpy(target.astype(numpy.float32))
    weight_tensor = torch.from_numpy(weight.astype(numpy.float32))
    positive = torch.tensor(positive_weight)

    def loss_of(rows):
        logits = head(feature_tensor[rows])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_tensor[rows], reduction = "none", pos_weight = positive)
        return (loss * weight_tensor[rows]).mean()

    best_loss, best_state, without_gain = math.inf, None, 0
    for epoch in range(30):
        head.train()
        order = generator.permutation(train_rows)
        for start in range(0, len(order), 256):
            loss = loss_of(order[start:start + 256])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        head.eval()
        with torch.no_grad():
            validation_loss = float(loss_of(validation_rows))
        if validation_loss < best_loss - 1e-4:
            best_loss, best_state, without_gain = validation_loss, {key: value.clone() for key, value in head.state_dict().items()}, 0
        else:
            without_gain += 1
            if without_gain >= 5:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head


def probabilities(head, features, rows):
    with torch.no_grad():
        return torch.sigmoid(torch.cat([head(torch.from_numpy(features[rows[start:start + 4096]])) for start in range(0, len(rows), 4096)])).numpy()


def validation_mask(frames, train_mask, seed):
    generator = numpy.random.default_rng(seed)
    is_validation = numpy.zeros(len(frames), dtype = bool)
    for sheet, group in frames[train_mask].groupby("sheet"):
        episodes = sorted(group["episode_id"].unique())
        generator.shuffle(episodes)
        chosen = episodes[:max(1, int(round(0.15 * len(episodes))))]
        is_validation |= frames["episode_id"].isin(chosen).to_numpy() & train_mask
    return is_validation


def main(variant_names) -> None:
    torch.set_num_threads(3)
    started = time.perf_counter()
    frames = audit_cleaned(read_frames())
    view = pandas.read_csv(AUDIT_FOLDER / "view_and_tracks" / "s09_view_rows.csv")
    frames = frames.merge(view[["frame_index", "line_share_top_half"]], on = "frame_index", how = "left")
    frames["no_line_ahead"] = frames["line_share_top_half"].fillna(1.0) <= 0.002
    frames["goal_distance_cm"] = frames["goal_distance_cm"].fillna(0.0)
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    all_features = numpy.load(PROCESSED_FOLDER / "features" / "resnet18_imagenet_grid.npy", mmap_mode = "r")
    features = numpy.asarray(all_features[frames["frame_index"].to_numpy()], dtype = numpy.float16)
    is_stop = (frames["action"] == "stop").to_numpy()
    near_goal = (frames["goal_distance_cm"] <= 2.0).to_numpy()
    far = (frames["goal_distance_cm"] > 3.0).to_numpy() & ~is_stop
    no_line = frames["no_line_ahead"].to_numpy()
    say(f"rows {len(frames)}, stop rows {int(is_stop.sum())}, within 2 cm {int(near_goal.sum())}, far non-stop rows with no line ahead "
        f"{int((far & no_line).sum())} of {int(far.sum())}")
    variants = {
        "A": {"target": "stop", "hard": 0, "seeds": [1, 2, 3]},
        "B5": {"target": "stop", "hard": 5, "seeds": [1, 2, 3]},
        "B10": {"target": "stop", "hard": 10, "seeds": [1]},
        "C": {"target": "near", "hard": 0, "seeds": [1]},
        "D": {"target": "near", "hard": 5, "seeds": [1]},
        "E": {"target": "stop", "hard": "mined", "seeds": [1, 2, 3]}
    }
    if variant_names:
        variants = {name: settings for name, settings in variants.items() if name in variant_names}
    sheets = sorted(frames["sheet"].unique())
    columns = ["frame_index", "episode_id", "sheet", "row_in_episode", "goal_distance_cm", "step_fraction", "no_line_ahead", "action"]
    summary = []
    for name, settings in variants.items():
        for seed in settings["seeds"]:
            target = is_stop if settings["target"] == "stop" else near_goal
            positive_weight = 10.0 if settings["target"] == "stop" else 3.0
            tables = {}
            for fold_number, sheet in enumerate(sheets):
                test_mask = (frames["sheet"] == sheet).to_numpy()
                train_mask = ~test_mask
                is_validation = validation_mask(frames, train_mask, 9_11_26 + fold_number)
                train_rows = numpy.flatnonzero(train_mask & ~is_validation)
                validation_rows = numpy.flatnonzero(is_validation)
                weight = numpy.ones(len(frames))
                if isinstance(settings["hard"], (int, float)) and settings["hard"] > 0:
                    weight[far & no_line] = settings["hard"]
                head = train_head(features, target, weight, positive_weight, train_rows, validation_rows, seed * 100 + fold_number)
                if settings["hard"] == "mined":
                    train_probability = probabilities(head, features, numpy.flatnonzero(train_mask))
                    mined = numpy.zeros(len(frames), dtype = bool)
                    mined[numpy.flatnonzero(train_mask)] = train_probability >= 0.2
                    weight[far & mined] = 5.0
                    head = train_head(features, target, weight, positive_weight, train_rows, validation_rows, seed * 100 + fold_number)
                test_rows = numpy.flatnonzero(test_mask)
                table = frames.loc[test_rows, columns].reset_index(drop = True)
                table["stop_probability"] = probabilities(head, features, test_rows)
                tables[sheet] = table
            grid = {(sheet, consecutive, threshold): episode_outcomes(table, threshold, consecutive)
                    for sheet, table in tables.items() for consecutive in (1, 2) for threshold in THRESHOLDS}
            for consecutive in (1, 2):
                chosen_rows, best_rows = [], []
                for sheet in sheets:
                    others = [other for other in sheets if other != sheet]
                    def pooled(threshold):
                        return shares(pandas.concat([grid[(other, consecutive, threshold)] for other in others]))
                    chosen = max(THRESHOLDS, key = lambda threshold: (pooled(threshold)["right"], -pooled(threshold)["early"]))
                    chosen_rows.append((sheet, chosen, shares(grid[(sheet, consecutive, chosen)])))
                    best = max(THRESHOLDS, key = lambda threshold: (shares(grid[(sheet, consecutive, threshold)])["right"], -shares(grid[(sheet, consecutive, threshold)])["early"]))
                    best_rows.append((sheet, best, shares(grid[(sheet, consecutive, best)])))
                for rule, rows in (("chosen on other sheets", chosen_rows), ("best on own sheet", best_rows)):
                    right = [row[2]["right"] for row in rows]
                    early = [row[2]["early"] for row in rows]
                    never = [row[2]["never"] for row in rows]
                    say(f"{name} seed {seed}, {consecutive} in a row, {rule}: right {numpy.mean(right):.3f} ({min(right):.2f} to {max(right):.2f}), "
                        f"early {numpy.mean(early):.3f} ({min(early):.2f} to {max(early):.2f}), never {numpy.mean(never):.3f}; thresholds {[row[1] for row in rows]}; "
                        + " ".join(f"{row[0][-2:]}:{row[2]['right']:.2f}/{row[2]['early']:.2f}" for row in rows))
                    summary.append({"variant": name, "seed": seed, "consecutive": consecutive, "rule": rule, "right_mean": round(float(numpy.mean(right)), 4),
                                    "right_min": min(right), "right_max": max(right), "early_mean": round(float(numpy.mean(early)), 4),
                                    "early_min": min(early), "early_max": max(early), "never_mean": round(float(numpy.mean(never)), 4),
                                    "thresholds": json.dumps([row[1] for row in rows])})
            pooled_table = pandas.concat(tables.values())
            far_table = pooled_table[pooled_table["goal_distance_cm"] > 5.0]
            say(f"{name} seed {seed}: stop probability over 5 cm from the goal, p90 with no line ahead {far_table.loc[far_table['no_line_ahead'], 'stop_probability'].quantile(0.9):.3f} "
                f"against {far_table.loc[~far_table['no_line_ahead'], 'stop_probability'].quantile(0.9):.3f} with a line; at the stop pictures median "
                f"{pooled_table.loc[pooled_table['action'] == 'stop', 'stop_probability'].median():.3f} ({time.perf_counter() - started:.0f} s so far)")
            pandas.DataFrame(summary).to_csv(OUTPUT_FOLDER / "stop_probe_summary.csv", index = False)
            (OUTPUT_FOLDER / "stop_probe.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
