"""
Frozen-feature probes of the exact-step design on the audit-cleaned rows (read only; results in this folder).

Question: the exact-step models almost never predict a backward move (within 45 deg on 0.27 of backward moves against
0.61 on the others, six held-out sheets), while the move-name models, trained with square-root class weights, get 0.40.
Does weighting the exact-step loss by the same square-root class weights of the demonstrated move name bring the
backward moves back, and at what cost to the forward ones? Head as in training_probes (one hidden layer of 256,
dropout 0.2, AdamW, Huber loss on cm and 5-deg units, stop output with positive weight 10 on the 259 stop pictures),
leave-one-sheet-out, early stopping on 15 % of each training sheet's episodes.
Variants: "plain" (as the final candidate's design) and "weighted" (row weight = square-root inverse frequency of the
row's move name, small motions and turns as one group, stop as one group, mean 1). Two seeds each.
Writes exact_probe.out.txt.
"""

import math
import sys

import numpy
import pandas
import torch

from common_frames import DIRECTION_NAMES, OUTPUT_FOLDER, PROCESSED_FOLDER, audit_cleaned, read_frames
from stop_probe import validation_mask

NOMINAL = {"forward": 0.0, "forward_left": 60.0, "left": 90.0, "back_left": 135.0, "back": 180.0, "back_right": -135.0,
           "right": -90.0, "forward_right": -60.0}
BACK = ["back", "back_left", "back_right"]
lines = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text, flush = True)


class ExactHead(torch.nn.Module):

    def __init__(self, mean, spread):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype = torch.float32))
        self.register_buffer("spread", torch.as_tensor(spread, dtype = torch.float32))
        self.shared = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(len(mean), 256), torch.nn.GELU(), torch.nn.Dropout(0.2))
        self.exact = torch.nn.Linear(256, 5)
        self.stop = torch.nn.Linear(256, 1)

    def forward(self, features):
        shared = self.shared((features.float() - self.mean) / self.spread)
        return self.exact(shared), self.stop(shared)[:, 0]


def nearest(angles):
    names = list(NOMINAL)
    values = numpy.array([NOMINAL[name] for name in names])
    differences = numpy.abs((numpy.asarray(angles)[:, None] - values[None, :] + 180.0) % 360.0 - 180.0)
    return numpy.array(names)[numpy.argmin(differences, axis = 1)]


def path_angles(frames):
    """Direction from the tip now to the tip 3 rows later (same episode), in the current picture's frame, and its length."""

    heading = numpy.radians(frames["picture_heading_deg"].to_numpy())
    tip_x = frames["tip_x_m"].to_numpy()
    tip_y = frames["tip_y_m"].to_numpy()
    episode = frames["episode_id"].to_numpy()
    angle = numpy.full(len(frames), numpy.nan)
    length = numpy.zeros(len(frames))
    for position in range(len(frames)):
        future = position
        for step in range(1, 4):
            if position + step < len(frames) and episode[position + step] == episode[position]:
                future = position + step
        moved_x, moved_y = tip_x[future] - tip_x[position], tip_y[future] - tip_y[position]
        forward = moved_x * math.cos(heading[position]) + moved_y * math.sin(heading[position])
        left = moved_x * math.cos(heading[position] + math.pi / 2) + moved_y * math.sin(heading[position] + math.pi / 2)
        angle[position] = math.degrees(math.atan2(left, forward))
        length[position] = 100.0 * math.hypot(forward, left)
    return angle, length


def main(seeds) -> None:
    torch.set_num_threads(3)
    frames = audit_cleaned(read_frames()).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    all_features = numpy.load(PROCESSED_FOLDER / "features" / "resnet18_imagenet_grid.npy", mmap_mode = "r")
    features = numpy.asarray(all_features[frames["frame_index"].to_numpy()], dtype = numpy.float16)
    exact = numpy.stack([100.0 * frames["tip_forward_m"].fillna(0.0), 100.0 * frames["tip_left_m"].fillna(0.0), frames["forearm_change_deg"].fillna(0.0),
                         frames["lean_change_deg"].fillna(0.0), frames["hand_change_deg"].fillna(0.0)], axis = 1).astype(numpy.float32)
    is_stop = (frames["action"] == "stop").to_numpy().astype(numpy.float32)
    group = frames["action"].where(frames["action"].isin(DIRECTION_NAMES + ["stop"]), "twist").to_numpy()
    counts = pandas.Series(group).value_counts()
    class_weight = numpy.array([math.sqrt(len(group) / counts[name]) for name in group])
    class_weight = class_weight / class_weight.mean()
    say("square-root weights by group: " + ", ".join(f"{name} {class_weight[group == name][0]:.2f}" for name in counts.index))
    moving = frames["action"].isin(DIRECTION_NAMES).to_numpy()
    true_class = nearest(frames["move_angle_deg"].to_numpy())
    is_back = numpy.isin(true_class, BACK) & moving
    path_angle, path_length = path_angles(frames)
    is_path = (frames["action"] != "stop").to_numpy() & (path_length >= 1.5)
    sheets = sorted(frames["sheet"].unique())
    scale = torch.tensor([1.0, 1.0, 5.0, 5.0, 5.0])
    for variant in ("plain", "weighted"):
        for seed in seeds:
            fold_rows = []
            for fold_number, sheet in enumerate(sheets):
                test_mask = (frames["sheet"] == sheet).to_numpy()
                is_validation = validation_mask(frames, ~test_mask, 9_11_26 + fold_number)
                train_rows = numpy.flatnonzero(~test_mask & ~is_validation)
                validation_rows = numpy.flatnonzero(is_validation)
                weight = class_weight if variant == "weighted" else numpy.ones(len(frames))
                torch.manual_seed(seed * 100 + fold_number)
                generator = numpy.random.default_rng(seed * 100 + fold_number)
                sample = features[train_rows[::max(1, len(train_rows) // 4000)]].astype(numpy.float64)
                head = ExactHead(sample.mean(axis = 0), sample.std(axis = 0) + 1e-3)
                optimiser = torch.optim.AdamW(head.parameters(), lr = 1e-3, weight_decay = 1e-2)
                feature_tensor = torch.from_numpy(features)
                exact_tensor = torch.from_numpy(exact)
                stop_tensor = torch.from_numpy(is_stop)
                weight_tensor = torch.from_numpy(weight.astype(numpy.float32))

                def loss_of(rows):
                    predicted_exact, stop_logit = head(feature_tensor[rows])
                    exact_loss = torch.nn.functional.huber_loss(predicted_exact / scale, exact_tensor[rows] / scale, reduction = "none", delta = 1.0).sum(dim = 1)
                    stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(stop_logit, stop_tensor[rows], reduction = "none", pos_weight = torch.tensor(10.0))
                    return (exact_loss * weight_tensor[rows]).mean() + stop_loss.mean()

                best_loss, best_state, without_gain = math.inf, None, 0
                for epoch in range(40):
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
                test_rows = numpy.flatnonzero(test_mask)
                with torch.no_grad():
                    predicted, _ = head(feature_tensor[test_rows])
                predicted = predicted.numpy()
                angle = numpy.degrees(numpy.arctan2(predicted[:, 1], predicted[:, 0]))
                error = numpy.abs((angle - frames["move_angle_deg"].to_numpy()[test_rows] + 180.0) % 360.0 - 180.0)
                path_error = numpy.abs((angle - path_angle[test_rows] + 180.0) % 360.0 - 180.0)
                test_moving = moving[test_rows]
                test_back = is_back[test_rows]
                test_path = is_path[test_rows]
                twist_true = exact[test_rows, 2]
                twisted = (numpy.abs(twist_true) >= 2.0) & (is_stop[test_rows] == 0)
                length = numpy.hypot(predicted[:, 0], predicted[:, 1])
                fold_rows.append({
                    "sheet": sheet,
                    "step45": float((error[test_moving] <= 45).mean()),
                    "path45": float((path_error[test_path] <= 45).mean()),
                    "back45": float((error[test_back] <= 45).mean()),
                    "other45": float((error[test_moving & ~test_back] <= 45).mean()),
                    "twist_sign": float((numpy.sign(predicted[twisted, 2]) == numpy.sign(twist_true[twisted])).mean()),
                    "length_median": float(numpy.median(length[test_moving])),
                    "predicted_back_share": float(numpy.isin(nearest(angle[test_moving]), ["back"]).mean())
                })
            table = pandas.DataFrame(fold_rows)
            say(f"{variant} seed {seed}: " + ", ".join(f"{column} {table[column].mean():.3f} ({table[column].min():.2f} to {table[column].max():.2f})"
                                                      for column in ["step45", "path45", "back45", "other45", "twist_sign", "length_median", "predicted_back_share"]))
            say("   per sheet step45/back45: " + " ".join(f"{row['sheet'][-2:]}:{row['step45']:.3f}/{row['back45']:.3f}" for _, row in table.iterrows()))
            (OUTPUT_FOLDER / "exact_probe.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main([int(value) for value in sys.argv[1:]] or [1, 2])
