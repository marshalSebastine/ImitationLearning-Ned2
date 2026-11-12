"""
Verifier's check of what a length gain does to the turning: forearm twist per cm of travel, asked by the models
against demonstrated, per kind of place, for the fold models on unseen sheets (cache of v05) and for model E on its
own training pictures (cache of v06). A length gain above 1 lowers the twist per cm by the same factor.

Read-only. Prints to the screen and to v13_twist_per_cm_held_out.out.txt.
"""

from pathlib import Path

import numpy

import verify_common

if __name__ == "__main__":
    frames = verify_common.demonstration_rows()
    cache_folder = verify_common.VERIFY_PARAMETERS["cache_folder"]
    moving = ~frames["is_stop"].to_numpy()
    lines = []
    for label, file_name in [("fold models on the sheet they never saw (recipe E)", "2026-11-10_resnet18_chunk_stop_weight_25.npz"),
                             ("model E (all sheets) on its own training pictures", "all_sheets_e_demonstrations.npz")]:
        outputs = numpy.load(cache_folder / file_name)
        assert numpy.array_equal(outputs["frame_index"], frames["frame_index"].to_numpy())
        asked_cm = numpy.linalg.norm(outputs["exact"][:, :2], axis = 1)
        asked_twist = numpy.abs(outputs["exact"][:, 2])
        lines.append(f"== {label} ==")
        for place in verify_common.PLACES + ["all places"]:
            chosen = moving & ((frames["place"] != "no place").to_numpy() if place == "all places" else (frames["place"] == place).to_numpy())
            demonstrated = frames.loc[chosen, "twist_deg"].abs().sum() / frames.loc[chosen, "step_cm"].sum()
            asked = asked_twist[chosen].sum() / asked_cm[chosen].sum()
            fixed = asked_twist[chosen].sum() / float(chosen.sum())
            lines.append(f"   {place:12s}: twist per cm: demonstrated {demonstrated:.2f} deg | asked per asked cm {asked:.2f} ({100 * asked / demonstrated:.0f} %) | "
                         f"asked per cm with the fixed 1 cm step {fixed:.2f} ({100 * fixed / demonstrated:.0f} %) | with a length gain of 1.3: {asked / 1.3:.2f} "
                         f"({100 * asked / 1.3 / demonstrated:.0f} %)")
    print("\n".join(lines))
    (Path(__file__).resolve().parent / "v13_twist_per_cm_held_out.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")
