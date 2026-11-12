"""
Verifier check 16: do the other all-sheets models ask for the walk round the puck on the pictures of runs 7 and 8?

Models: E (drove both runs), A (model 1), the team of five (B: A's recipe, seeds 1 to 5, answers averaged) and the
chunk model without the heavier stop weight (model 2). Each answers the recorded pictures of run 8 (sheet 06) and run 7
(sheet 05). Same pictures for every model, so this compares the models on the states E's driving produced; it does not
say where each model would have driven itself.

Run:  D:/AI_Project/.venv/Scripts/python.exe v16_other_models_on_the_runs.py > v16_other_models_on_the_runs.out.txt
About one minute. Nothing connects to the robot.
"""

import numpy
import pandas
import torch

from verify_common import ROOT_FOLDER, VERIFY_FOLDER, VERIFY_PARAMETERS

from model_mode import ModelPolicy  # noqa: E402

torch.set_num_threads(2)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 200)
RUNS = ROOT_FOLDER / "04_runs" / "gpu_ki014"
team = [RUNS / f"2026-11-10_resnet18_model_1_all_sheets{suffix}" / "checkpoints" / "all_sheets.pt" for suffix in ("", "_seed_2", "_seed_3", "_seed_4", "_seed_5")]
MODELS = {
    "E": ("resnet18_model_4_chunk_stop_25_all_sheets", RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "checkpoints" / "all_sheets.pt",
          RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "metrics.json"),
    "A": ("resnet18_model_1_all_sheets", team[0], RUNS / "2026-11-10_resnet18_model_1_all_sheets" / "metrics.json"),
    "B": ("resnet18_model_1_all_sheets", team, RUNS / "2026-11-10_resnet18_model_1_all_sheets" / "metrics.json"),
    "chunk": ("resnet18_model_2_chunk_all_sheets", RUNS / "2026-11-10_resnet18_model_2_chunk_all_sheets" / "checkpoints" / "all_sheets.pt",
              RUNS / "2026-11-10_resnet18_model_2_chunk_all_sheets" / "metrics.json")
}
policies = {}
for name, (run_name, checkpoint, metrics) in MODELS.items():
    try:
        policies[name] = ModelPolicy(run_name, checkpoint, metrics)
    except Exception as problem:
        print(f"model {name} not loaded: {problem}")
torch.set_num_threads(2)

for run_label, key, zone in (("run 8 (sheet 06)", "run_8_folder", (27, 40)), ("run 7 (sheet 05)", "run_7_folder", (27, 62))):
    folder = VERIFY_PARAMETERS["model_runs_folder"] / VERIFY_PARAMETERS[key]
    steps = pandas.read_csv(folder / "steps.csv")
    table = pandas.DataFrame({"step": steps["step"], "joint_4": numpy.degrees(steps["joint_4"]).round(0)})
    for name, policy in policies.items():
        answers = [policy.decide(open(folder / frame, "rb").read()) for frame in steps["frame"]]
        table[f"{name}_angle"] = [answer["angle_deg"] for answer in answers]
        table[f"{name}_len"] = [numpy.hypot(answer["tip_forward_cm"], answer["tip_left_cm"]) for answer in answers]
        table[f"{name}_twist"] = [answer["twist_deg"] for answer in answers]
        table[f"{name}_stop"] = [answer["stop_probability"] for answer in answers]
    table.round(3).to_csv(VERIFY_FOLDER / f"v16_other_models_{key[:5]}.csv", index = False)
    print(f"\n===== {run_label}: every model on the recorded pictures =====")
    inside = table[(table["step"] >= zone[0]) & (table["step"] <= zone[1])]
    print(inside.round(1).to_string(index = False))
    print(f"steps {zone[0]} to {zone[1]} (where the walk round the puck belongs):")
    for name in policies:
        gap = numpy.abs((table[f"{name}_angle"] - table["E_angle"] + 180.0) % 360.0 - 180.0)
        print(f"  {name}: twist asked summed {inside[f'{name}_twist'].sum():+.1f} deg; answers beyond 90 deg: {int((inside[f'{name}_angle'].abs() > 90).sum())} of {len(inside)};"
              f" asked length median {inside[f'{name}_len'].median():.2f} cm; first step with a stop chance of 0.5 or more on two pictures in a row:"
              f" {next((int(table['step'].iloc[position]) for position in range(1, len(table)) if table[f'{name}_stop'].iloc[position] >= 0.5 and table[f'{name}_stop'].iloc[position - 1] >= 0.5), None)};"
              f" direction differs from E's by {gap.mean():.1f} deg on average over the whole run")
