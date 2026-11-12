"""
The verifier's own read-out of the models' picture descriptors and answers (independent of the analyst's
extract_features.py: the descriptors are caught with forward hooks while the model runs as a whole, and the robot
pictures go through the robot driver's own ``ModelPolicy.picture_input`` and ``ModelPolicy.decide``).

Parts:
    model_e          model E (all sheets): every robot picture of the ten runs and all 18,522 training pictures;
                     also checks ``policy.decide`` against the answers logged on the robot (runs 2 to 9).
    fold <k>         the fold model of model E's recipe trained WITHOUT sheet k (1 to 6): only the demonstration
                     pictures of sheet k and the robot pictures taken on sheet k.

Nothing connects to the robot. Run from this folder (two to three minutes per part on two threads):
    D:/AI_Project/.venv/Scripts/python.exe extract_own.py model_e
    D:/AI_Project/.venv/Scripts/python.exe extract_own.py fold 3
Output: tables in the verifier's cache folder (verify_common.VERIFY_PARAMETERS["own_cache_folder"]) and
extract_own_<part>.out.txt here.
"""

import sys
import time

import numpy
import torch

from verify_common import (RUN_FOLDERS, RUN_MODELS, RUN_SHEETS, VERIFY_FOLDER, VERIFY_PARAMETERS, Report, run_log, run_picture_path,
                           training_frames)

EXTRACT_PARAMETERS = {
    "model_e_folder": "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets",
    "model_e_run_name": "resnet18_model_4_chunk_stop_25_all_sheets",
    "fold_folder": "2026-11-10_resnet18_chunk_stop_weight_25",
    "fold_run_name": "resnet18_chunk_stop_weight_25",
    "batch_size": 128
}



class HookedModel:

    def __init__(self, policy):
        """
        Wrap the policy's network so that every forward pass also yields what the prediction head received.

        Parameters
        ----------
        policy : model_mode.ModelPolicy
            The loaded model; ``policy.models[0]`` is hooked.
        """

        self.policy = policy
        self.model = policy.models[0]
        self.caught = {}
        self.model.head.register_forward_hook(self.keep_head_input)
        self.model.head.shared.register_forward_hook(self.keep_hidden)


    def keep_head_input(self, module, inputs, output) -> None:
        """Keep the pooled backbone features handed to the head (before the head standardises them)."""

        self.caught["grid"] = inputs[0].detach().float().numpy().copy()


    def keep_hidden(self, module, inputs, output) -> None:
        """Keep the head's shared hidden layer."""

        self.caught["hidden"] = output.detach().float().numpy().copy()


    def answers(self, network_input: torch.Tensor) -> dict:
        """
        Run the whole network on prepared input.

        Parameters
        ----------
        network_input : torch.Tensor
            (batch, 3, height, width), as ``ModelPolicy.picture_input`` or ``augmented_pictures`` returns it.

        Returns
        -------
        dict of str to numpy.ndarray
            "grid", "hidden", "exact", "stop_probability" and "chunk" (when the design has one).
        """

        from training_probes import predictions_from_outputs

        with torch.inference_mode():
            outputs = self.model(network_input)
            predictions = predictions_from_outputs(outputs, self.policy.class_names, self.policy.settings["stop_target"])
        result = {"grid": self.caught["grid"], "hidden": self.caught["hidden"], "exact": outputs["exact"].numpy().copy(),
                  "stop_probability": numpy.asarray(predictions["stop_probability"], dtype = numpy.float32)}
        if "chunk" in outputs:
            result["chunk"] = outputs["chunk"].numpy().copy()
        return result



def stacked(parts: list) -> dict:
    """Join a list of per-batch dictionaries into one dictionary of arrays."""

    return {name: numpy.concatenate([part[name] for part in parts]) for name in parts[0]}



def robot_answers(hooked: HookedModel, run_numbers: list) -> dict:
    """Run the hooked model on every logged robot JPEG of the named runs, one picture at a time, as the driver does."""

    parts, runs, steps = [], [], []
    for run_number in run_numbers:
        for step in range(len(run_log(run_number))):
            parts.append(hooked.answers(hooked.policy.picture_input(run_picture_path(run_number, step).read_bytes())))
            runs.append(run_number)
            steps.append(step)
    result = stacked(parts)
    result["run"], result["step"] = numpy.array(runs), numpy.array(steps)
    return result



def cache_answers(hooked: HookedModel, frame_indexes: numpy.ndarray, report: Report) -> dict:
    """Run the hooked model on pictures of the 160 x 120 training cache, in batches."""

    from train_policy import augmented_pictures

    pictures = numpy.load(VERIFY_PARAMETERS["processed_folder"] / "images_wide_160.npy", mmap_mode = "r")
    parts = []
    size = EXTRACT_PARAMETERS["batch_size"]
    for start in range(0, len(frame_indexes), size):
        batch = torch.from_numpy(numpy.array(pictures[frame_indexes[start:start + size]]))
        network_input, _ = augmented_pictures(batch, hooked.policy.settings, numpy.random.default_rng(0), is_training = False)
        parts.append(hooked.answers(network_input))
        if (start // size) % 30 == 0:
            report.say(f"    training pictures {start} of {len(frame_indexes)}")
    result = stacked(parts)
    result["frame_index"] = numpy.asarray(frame_indexes)
    return result



def save_table(name: str, table: dict) -> None:
    """Save a table to the verifier's cache folder; the wide "grid" table in half precision."""

    folder = VERIFY_PARAMETERS["own_cache_folder"]
    folder.mkdir(parents = True, exist_ok = True)
    arrays = dict(table)
    arrays["grid"] = arrays["grid"].astype(numpy.float16)
    numpy.savez(folder / f"{name}.npz", **arrays)



def main() -> None:

    from model_mode import ModelPolicy

    part = sys.argv[1]
    models = VERIFY_PARAMETERS["trained_models_folder"]
    started = time.perf_counter()
    if part == "model_e":
        report = Report(VERIFY_FOLDER / "extract_own_model_e.out.txt")
        folder = models / EXTRACT_PARAMETERS["model_e_folder"]
        policy = ModelPolicy(EXTRACT_PARAMETERS["model_e_run_name"], folder / "checkpoints" / "all_sheets.pt", folder / "metrics.json")
        torch.set_num_threads(VERIFY_PARAMETERS["threads"])
        hooked = HookedModel(policy)
        report.say(f"model E: {folder / 'checkpoints' / 'all_sheets.pt'}")

        report.say("")
        report.say("CHECK B: policy.decide on the saved JPEG against the answer logged on the robot (largest absolute difference per run)")
        report.say("run  model  pictures  forward cm  left cm   twist deg  stop chance")
        total = 0
        pairs = (("tip_forward_cm", "model_tip_forward_cm"), ("tip_left_cm", "model_tip_left_cm"),
                 ("twist_deg", "model_twist_deg"), ("stop_probability", "model_stop_probability"))
        for run_number in RUN_FOLDERS:
            if RUN_MODELS[run_number] != "E":
                continue
            log = run_log(run_number)
            decisions = [policy.decide(run_picture_path(run_number, step).read_bytes()) for step in range(len(log))]
            differences = [numpy.abs(numpy.array([decision[key] for decision in decisions]) - log[column].to_numpy()).max()
                           for key, column in pairs]
            total += len(log)
            report.say(f"{run_number:3d}  E      {len(log):8d}  {differences[0]:.6f}    {differences[1]:.6f}  {differences[2]:.6f}   {differences[3]:.6f}")
        report.say(f"pictures of model E's runs checked: {total}")

        robot = robot_answers(hooked, list(RUN_FOLDERS))
        save_table("model_e_robot", robot)
        report.say(f"robot pictures: {len(robot['run'])}; grid {robot['grid'].shape[1]}, hidden {robot['hidden'].shape[1]}")
        frames = training_frames()
        training = cache_answers(hooked, frames["frame_index"].to_numpy(), report)
        save_table("model_e_training", training)
        report.say(f"training pictures: {len(training['frame_index'])}")
    else:
        sheet_number = int(sys.argv[2])
        sheet = f"path_{sheet_number:02d}"
        report = Report(VERIFY_FOLDER / f"extract_own_fold_{sheet_number}.out.txt")
        folder = models / EXTRACT_PARAMETERS["fold_folder"]
        policy = ModelPolicy(EXTRACT_PARAMETERS["fold_run_name"], folder / "checkpoints" / f"without_{sheet}.pt", folder / "metrics.json")
        torch.set_num_threads(VERIFY_PARAMETERS["threads"])
        hooked = HookedModel(policy)
        report.say(f"fold model without {sheet}: {folder / 'checkpoints' / f'without_{sheet}.pt'}")
        frames = training_frames()
        frame_indexes = frames.loc[frames["sheet"] == sheet, "frame_index"].to_numpy()
        table = cache_answers(hooked, frame_indexes, report)
        save_table(f"fold_without_{sheet}_sheet", table)
        runs_here = [run_number for run_number, run_sheet in RUN_SHEETS.items() if run_sheet == sheet]
        robot = robot_answers(hooked, runs_here)
        save_table(f"fold_without_{sheet}_robot", robot)
        report.say(f"{len(frame_indexes)} demonstration pictures of {sheet} and {len(robot['run'])} robot pictures (runs {runs_here})")
    report.say(f"done after {time.perf_counter() - started:.0f} s")
    report.save()



if __name__ == "__main__":
    main()
