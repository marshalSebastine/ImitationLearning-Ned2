"""
Robot hours for the improvement options, calculated from the measured times in episode_timing.out.txt (this folder)
and the data-amount fit in data_amount_regression.out.txt. Every guess is a named parameter below.

Usage: python robot_hours.py
"""

PARAMETERS = {
    # measured (episode_timing.out.txt, 210 kept demonstrations, 239 hand corrections)
    "seconds_per_step": 1.241,
    "setup_before_first_row_min": 0.75,
    "last_row_to_finish_min": 0.19,
    "finish_to_next_start_min": 0.92,
    "take_over_min": 51.64 / 60.0,
    "demonstration_cycle_min": 5.78,
    "minutes_per_kept_episode_all_sittings": 8.13,
    "minutes_per_kept_episode_last_sitting": 5.64,
    "near_end_cycle_min": 2.40,
    "steps_per_sheet": {"01": 31.5, "02": 66.0, "03": 106.0, "04": 168.0, "05": 103.5, "06": 82.0},
    # measured (data_amount_regression.out.txt): points of within 45 and of stopped right (tuned, two in a row) per
    # doubling of training pictures and per doubling of training sheets (frozen ResNet-18 grid probe, classes head)
    "direction_per_picture_doubling": 0.0170,
    "direction_per_sheet_doubling": 0.0251,
    "stop_per_picture_doubling": 0.0865,
    "stop_per_sheet_doubling": 0.0327,
    "current_rows": 18053,
    "current_sheets": 6,
    # guesses (judgement, to be replaced by the pilot's measurements)
    "model_step_factor": (1.0, 1.5),
    "sitting_overhead_factor": (1.0, 8.13 / 5.78),
    "base_success": 0.3,
    "new_track_steps": 92.8,
    "new_track_setup_h": 0.5,
    "episodes_per_new_track": 20,
}


def trial_minutes(steps, parameters, factor):
    """
    One closed-loop model run: set-up, the model's steps, the end and the reset.
    """
    return (parameters["setup_before_first_row_min"] + steps * factor * parameters["seconds_per_step"] / 60.0
            + parameters["last_row_to_finish_min"] + parameters["finish_to_next_start_min"])


def main(parameters = None):
    """
    Print the hours of each option, low and high.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    low_factor, high_factor = settings["model_step_factor"]
    low_overhead, high_overhead = settings["sitting_overhead_factor"]
    sheets = settings["steps_per_sheet"]
    print("=== one model run (trial), minutes, per sheet (step factor 1.0 to 1.5, no sitting overhead) ===")
    for sheet, steps in sheets.items():
        print(f"sheet {sheet}: {trial_minutes(steps, settings, low_factor):.1f} to {trial_minutes(steps, settings, high_factor):.1f}")
    six_low = sum(trial_minutes(steps, settings, low_factor) for steps in sheets.values())
    six_high = sum(trial_minutes(steps, settings, high_factor) for steps in sheets.values())
    print(f"one trial on each of the six sheets: {six_low:.1f} to {six_high:.1f} min")
    for trials in (5, 10):
        print(f"{trials} trials per sheet: {trials * six_low / 60:.1f} to {trials * six_high / 60 * high_overhead:.1f} h "
              f"(high end with the sitting overhead)")
    new_low = trial_minutes(settings["new_track_steps"], settings, low_factor)
    new_high = trial_minutes(settings["new_track_steps"], settings, high_factor)
    for tracks, trials in ((10, 2), (10, 3)):
        print(f"{tracks} sealed new tracks x {trials} trials: {tracks * trials * new_low / 60:.1f} to "
              f"{tracks * trials * new_high / 60 * high_overhead:.1f} h")
    print()
    print("=== correction round (one take-over per failed run, the person finishes the track) ===")
    failure = 1.0 - settings["base_success"]
    episode_low = new_low + failure * settings["take_over_min"]
    episode_high = new_high + failure * settings["take_over_min"]
    print(f"per episode: {episode_low:.1f} to {episode_high * high_overhead:.1f} min (base success {settings['base_success']})")
    for episodes in (5, 30, 50, 100):
        corrections = episodes * failure
        rows = corrections * settings["new_track_steps"] / 2.0
        print(f"{episodes} episodes: {episodes * episode_low / 60:.1f} to {episodes * episode_high * high_overhead / 60:.1f} h; "
              f"about {corrections:.0f} corrections and {rows:.0f} correction rows if failures come at mid-track "
              f"({100 * rows / settings['current_rows']:.0f} percent of the {settings['current_rows']} rows)")
    print()
    print("=== demonstrations (measured whole-sitting rates: last sitting to all sittings) ===")
    for label, episodes in (("100 more on the six sheets", 100), ("210 more on the six sheets (doubling)", 210)):
        print(f"{label}: {episodes * settings['minutes_per_kept_episode_last_sitting'] / 60:.1f} to "
              f"{episodes * settings['minutes_per_kept_episode_all_sittings'] / 60:.1f} h")
    for tracks in (4, 6, 8):
        episodes = tracks * settings["episodes_per_new_track"]
        low = episodes * settings["minutes_per_kept_episode_last_sitting"] / 60 + tracks * settings["new_track_setup_h"]
        high = episodes * settings["minutes_per_kept_episode_all_sittings"] / 60 + tracks * settings["new_track_setup_h"]
        print(f"{tracks} new tracks x {settings['episodes_per_new_track']} episodes ({episodes}): {low:.1f} to {high:.1f} h")
    print(f"near-end starts for the stop: {60 / settings['near_end_cycle_min']:.0f} per hour at the median cycle")
    print()
    print("=== expected offline gain, frozen-probe fit (extrapolated beyond 5 training sheets: a guess) ===")
    import math
    rows_per_episode = settings["current_rows"] / 209.0
    for label, new_episodes, new_sheets in (("100 more episodes on the six sheets", 100, 0),
                                            ("210 more episodes on the six sheets", 210, 0),
                                            ("4 new tracks x 20 episodes", 80, 4),
                                            ("6 new tracks x 20 episodes", 120, 6),
                                            ("8 new tracks x 20 episodes", 160, 8)):
        rows = settings["current_rows"] + new_episodes * rows_per_episode
        picture_doublings = math.log2(rows / settings["current_rows"])
        sheet_doublings = math.log2((settings["current_sheets"] + new_sheets) / settings["current_sheets"])
        direction = (settings["direction_per_picture_doubling"] * picture_doublings
                     + settings["direction_per_sheet_doubling"] * sheet_doublings)
        stop = settings["stop_per_picture_doubling"] * picture_doublings + settings["stop_per_sheet_doubling"] * sheet_doublings
        print(f"{label}: pictures x{rows / settings['current_rows']:.2f}, sheets x{(settings['current_sheets'] + new_sheets) / settings['current_sheets']:.2f}; "
              f"within 45 {100 * direction:+.1f} points, stopped right {100 * stop:+.1f} points")


if __name__ == "__main__":
    main()
