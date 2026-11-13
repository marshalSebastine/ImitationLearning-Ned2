# Training overnight on the GPU pool

Written 13.11.26 for Marshal (works the same for Emranul). Nothing here touches the robot. The commands run in the
Mac's Terminal unless a step says **on the GPU machine**.

What one night does: a check, then the five policies three times each (seeds 1, 2, 3), then an Optuna search over the
first policy, then one results table. Each run stops by itself after 90 minutes at most; the Optuna search after
3 hours at most. Expect 4 to 7 hours in total, depending on the GPU (guessed; the GPU's model is not known yet).

## Before the first night (once)

1. **Access.** SSH must be activated by Prof. Nierhoff, and the OTH VPN (FortiClient) must be connected, even on
   eduroam. The university's guide is `01_course/gpu_pool_ssh_guide_en.pdf`; questions go to Tamara Graf, DC 1.01.
2. **Pick a machine** in DC 1.07 from the last row (usually the only ones left on at night). Its number is on its
   monitor: `ki-0XX`.
3. **Make a short name for it** on the Mac, so the `@` in the user name cannot go wrong. Open `~/.ssh/config` in a text
   editor and add, with your own OTH address and the machine's number:

       Host gpu
           HostName ki-0XX.oth-aw.de
           User first.last@oth-aw.de

   Then `ssh gpu` logs in. The first time it asks about the fingerprint: the question must name `ki-0XX.oth-aw.de`;
   if it names `oth-aw.de` alone, answer no (wrong computer, see the guide).
4. **Copy the project** (code, the main data and the dataset table; about 400 MB). From the project folder on the Mac:

       rsync -av --exclude '.venv/' --exclude '03_data/' --exclude '*.zip' ./ gpu:AI_Project/
       rsync -avR --exclude 'pictures_*' 03_data/raw/main_data 03_data/processed gpu:AI_Project/

5. **On the GPU machine** (`ssh gpu`), check the GPU, the free space and Python:

       nvidia-smi
       df -h ~
       python3 --version

   `nvidia-smi` must list a GPU, and no one else's program should be using it. About 8 GB free is needed (the packages
   take about 5 GB). Python 3.8 or newer is needed.
6. **On the GPU machine**, install the packages (10 to 20 minutes):

       cd ~/AI_Project
       python3 -m venv .venv_gpu
       .venv_gpu/bin/pip install --upgrade pip
       .venv_gpu/bin/pip install -r 02_code/requirements_training.txt

   If `python3 -m venv` says venv is missing, tell Claude the message: the machine needs another way.

## Each night

1. **On the GPU machine:** the check (3 to 5 minutes the first time, as it shrinks all pictures once):

       cd ~/AI_Project/02_code
       ../.venv_gpu/bin/python train_policy.py --check

   It must say `device cuda`, list the GPU, show `OK` for all five policies and end with `everything ready`.
   `device cpu` means the GPU is not used: stop and ask Claude.
2. **Start the night** inside `tmux`, so it keeps running when you log out. Choose the batch name once and write it down:

       tmux new -s night
       bash run_overnight.sh 2026-11-14_overnight

   When the first epochs are printed, press **Ctrl-B, then D** to leave tmux running. Then `exit` to log out.
   Without tmux: `nohup bash run_overnight.sh 2026-11-14_overnight > night.out 2>&1 &` instead.
3. **To look in during the night:** `ssh gpu`, then `tmux attach -t night` (leave again with Ctrl-B, D), or
   `tail -n 30 ~/AI_Project/04_runs/logs/2026-11-14_overnight.log`.
4. **If the machine was switched off or the night stopped:** start the same command with the **same batch name**. Finished
   runs are skipped and the Optuna search carries on from its last trial.

## The next morning

1. Read the table: `cat ~/AI_Project/04_runs/2026-11-14_overnight_summary.md` (and `..._optuna.md`).
2. Copy the results to the Mac (weights included, about 45 MB per run), from the project folder on the Mac:

       rsync -av gpu:AI_Project/04_runs/ 04_runs/

3. Send Claude the summary. The weights stay out of git (`04_runs/**/checkpoints/`); the summary, `metrics.json` and
   `history.csv` files are small and may be committed.

## How to read the table

- **Hard-sheet score**: on the validation episodes of sheets 03, 04 and 06 (the ones Emranul's models could not
  finish), the mean of the move's balanced accuracy and the forearm label's accuracy. Higher is better; 1.0 is perfect.
- **Spread over seeds**: if two policies differ by less than their spread, the table does not tell them apart.
- **Test scores**: for the report only. Choosing a policy by them would make them meaningless.
- These are offline numbers. Which policy really pushes the puck best is decided on the robot.

## Other commands

    ../.venv_gpu/bin/python train_policy.py --model history_classifier --seed 1 --batch 2026-11-14_extra
    ../.venv_gpu/bin/python train_policy.py --model factorised_classifier --held-out-sheet path_04 --batch 2026-11-14_held_out
    ../.venv_gpu/bin/python tune_policy.py --batch 2026-11-14_overnight --trials 50 --hours 5
    ../.venv_gpu/bin/python train_policy.py --summary --batch 2026-11-14_overnight

`--held-out-sheet` trains without one sheet and scores that sheet alone: the closest offline test of an unseen track.
Tests of the training code (CPU, about a minute): `../.venv_gpu/bin/python test_policy_training.py`.
