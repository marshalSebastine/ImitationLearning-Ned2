"""Review attack 1: --step-cm with odd values and together with --step-length predicted."""

import contextlib
import io

import safe_import
from safe_import import model_mode
from config import MODEL_MODE_PARAMETERS


def outcome(arguments):

    complaint = io.StringIO()
    try:
        with contextlib.redirect_stderr(complaint):
            return ("taken", model_mode.command_line(arguments)[1])
    except SystemExit as exit_:
        return (f"refused (exit {exit_.code})", complaint.getvalue().strip().splitlines()[-1][:230])


for value in ["0.65", "0.1", "1.0", "1", "0.10000000000000001", "0.09999999", "0.0999999999999999999", "1.0000001", "1.00000000000000001", "1e0", "1E-1",
              "+0.65", ".65", "0.65 ", " 0.65", "0,65", "0.65cm", "6.5", "65", "0", "-0", "-0.65", "nan", "NaN", "inf", "-inf", "infinity", "1_0", "0_.65", "0.6_5",
              "0x1p-1", "", "٠.٦٥", "0.123456789", "0.1000004", "0.9999996", "0.65e0", "6.5e-1", "65e-2", "１"]:
    print(repr(value).ljust(26), outcome(["simulated", "--step-cm", value]))

print()
print("with predicted:", outcome(["robot", "--step-cm", "0.65", "--step-length", "predicted"]))
print("with fixed:", outcome(["robot", "--step-cm", "0.65", "--step-length", "fixed"]))
print("with length gain only:", outcome(["robot", "--step-cm", "0.65", "--length-gain", "0.5"]))
print("equals form:", outcome(["robot", "--step-cm=0.65"]))
print("twice:", outcome(["robot", "--step-cm", "0.65", "--step-cm", "0.3"]))
print("twice, last bad:", outcome(["robot", "--step-cm", "0.65", "--step-cm", "3"]))
print("abbreviation --step 0.65:", outcome(["robot", "--step", "0.65"]))
print("abbreviation --step-c 0.65:", outcome(["robot", "--step-c", "0.65"]))
print("abbreviation --step-l predicted + --step-cm:", outcome(["robot", "--step-l", "predicted", "--step-cm", "0.5"]))
print("abbreviation --stal log:", outcome(["robot", "--stal", "log"]))
print("abbreviation --st log:", outcome(["robot", "--st", "log"]))
print("--stall LOG:", outcome(["robot", "--stall", "LOG"]))
print("--stall (none):", outcome(["robot", "--stall"]))
print("--ghost '':", outcome(["robot", "--ghost", ""]))
print("--run-folder '':", outcome(["robot", "--run-folder", ""]))
print("--label '' :", outcome(["robot", "--label", ""]), model_mode.command_line(["robot", "--label", ""])[0].label)
print("--step-budget 0:", outcome(["robot", "--step-budget", "0"]))
print("--step-budget -3:", outcome(["robot", "--step-budget", "-3"]))

# when config.py itself says "predicted": the flag alone must be refused, with --step-length fixed it must be taken
MODEL_MODE_PARAMETERS["step_length_rule"] = "predicted"
print()
print("config predicted, --step-cm alone:", outcome(["robot", "--step-cm", "0.65"]))
print("config predicted, with --step-length fixed:", outcome(["robot", "--step-cm", "0.65", "--step-length", "fixed"]))
MODEL_MODE_PARAMETERS["step_length_rule"] = "fixed"
