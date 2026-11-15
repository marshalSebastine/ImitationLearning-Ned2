# Check of "Action chunking and ACT" (key: action_chunking)

Sceptical check of the researcher's 49 claims and conclusion, done on 9 November 2026.
Every verdict below is **checked** (I opened the source and read the passage myself) unless it says **unchecked**.
Quotes are avoided on purpose; everything is paraphrased. "Calculated" means my own arithmetic, not a number from a paper.

Result in one line: 44 claims supported, 5 corrected (11, 15, 31, 37, 49), 0 not supported, 0 could not open; the conclusion also needs three wording fixes (section 3). The numbers are almost all right digit by digit. The weak point is the conclusion: it carries over high-frequency, multi-camera results to a one-camera, stop-and-go, CPU-trained setting with unseen tracks, and none of the sources tests that.

## 1. Sources I opened

| Short name | Authors, year, title, venue | URL opened | What I used |
|---|---|---|---|
| ACT | Zhao, Kumar, Levine, Finn (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. Robotics: Science and Systems XIX, paper 016, DOI 10.15607/RSS.2023.XIX.016 | https://arxiv.org/pdf/2304.13705 (text extracted locally from the PDF that WebFetch saved), https://arxiv.org/html/2304.13705v1, https://arxiv.org/html/2304.13705v1/fm_data.png (figure image, read visually), https://www.roboticsproceedings.org/rss19/p016.html | Sec. I, III, IV-A to IV-C, V-A to V-C, VI-A to VI-C, Tables I to III, Algorithms 1 and 2, Fig. 8 (PDF numbering) |
| ACT code | Zhao et al., github.com/tonyzhaozh/act (main branch) | raw.githubusercontent.com/tonyzhaozh/act/main/ imitate_episodes.py, detr/models/detr_vae.py, detr/models/transformer.py, detr/models/backbone.py, utils.py, README.md; https://github.com/tonyzhaozh/act/issues/25 | eval_bc, DETRVAE, Transformer, Backbone, EpisodicDataset, README tips |
| LeRobot ACT | Hugging Face LeRobot, main branch, opened 9 Nov 2026 | raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/act/configuration_act.py and modeling_act.py; https://huggingface.co/docs/lerobot/act; https://huggingface.co/docs/lerobot/main/hardware_guide.md | ACTConfig defaults and validate_features; ACT.__init__/forward; ACTTemporalEnsembler; compute guide |
| State-free | J. Zhao, Lu, D. Zhang, Liu, Liang, T. Zhang, Cao, Xie, Hu, S. Wang, Guo, D. Wang, Gao (2025). Do You Need Proprioceptive States in Visuomotor Policies? arXiv 2509.18644v2 (preprint, no venue) | https://arxiv.org/html/2509.18644v2, https://arxiv.org/abs/2509.18644v2 | Abstract, Sec. I, III-B, V-C, VIII, Tables III to V |
| Diffusion Policy | Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023; arXiv v5 (14 Mar 2024) is the extended journal version | https://arxiv.org/pdf/2303.04137v5 (text extracted locally), https://arxiv.org/html/2303.04137v5, https://arxiv.org/abs/2303.04137v5 | Sec. 4.3, 5.3, 6.1, Fig. 5, Tables 6 and 7, App. D.0.1 |
| VQ-BeT | Lee, Wang, Etukuru, Kim, Shafiullah, Pinto (2024). Behavior Generation with Latent Actions. ICML 2024, PMLR 235, pp. 26991 to 27008 | https://arxiv.org/html/2403.03181v2, https://proceedings.mlr.press/v235/lee24y.html | Sec. 4.6, 4.7, Table 11 |
| BAKU | Haldar, Peng, Pinto (2024). BAKU: An Efficient Transformer for Multi-Task Policy Learning. NeurIPS 2024 (poster) | https://arxiv.org/html/2406.07539v2, https://neurips.cc/virtual/2024/poster/93282 | Sec. 3.3, 4.4, Table 3, Table 4, App. A.3 |
| BID | Liu, Hamid, Xie, Lee, Du, Finn (2025). Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling. ICLR 2025 (poster) | https://arxiv.org/html/2408.17355v4, https://iclr.cc/virtual/2025/poster/28245 | Sec. 3, 5.1, 5.2, Tables 1 to 4, Table 6 |
| Chunking theory | T. Zhang, Pfrommer, Pan, Matni, Simchowitz (2025). Action Chunking and Exploratory Data Collection Yield Exponential Improvements in Behavior Cloning for Continuous Control. arXiv 2507.09061v5 (26 Nov 2025; no venue stated) | https://arxiv.org/html/2507.09061v5, https://arxiv.org/abs/2507.09061v5 | Sec. 3 to 5, Fig. 5, Fig. 10 (text only) |
| RTC | Black, Galliker, Levine (2025). Real-Time Execution of Action Chunking Flow Policies. NeurIPS 2025 (arXiv comment says published in NeurIPS 2025) | https://arxiv.org/html/2506.07339v2, https://arxiv.org/abs/2506.07339v2 | Abstract, Sec. 2, 4.1, 4.2 |
| OpenVLA-OFT | Kim, Finn, Liang (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. RSS 2025 | https://arxiv.org/html/2502.19645v2, https://arxiv.org/abs/2502.19645v2 | Sec. V, Tables I and II |
| SAIL | Ranawaka Arachchige, Chen, Jung, Shin, Bansal, Barroso, He, Lin, Joffe, Kousik, Xu (2025). SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies. CoRL 2025 | https://arxiv.org/html/2506.11948v2, https://arxiv.org/abs/2506.11948v2 | Table 1, App. B, Table K.7 |

All bibliographic details in the claims (authors, order, venues) match what I saw.

## 2. Verdicts, claim by claim

**1. supported.** ACT Sec. I and Sec. IV-A: the policy predicts target joint positions for the next k steps, written as pi(a_t:t+k | s_t) instead of pi(a_t | s_t); the authors say this reduces the effective horizon k-fold and mitigates compounding errors.

**2. supported.** Sec. I mentions temporally correlated confounders such as pauses that Markovian single-step policies struggle with; Sec. IV-A adds that chunking mitigates this when the confounder lies within a chunk, without the causal-confusion risk of history-conditioned policies. Argued, not measured on its own.

**3. supported.** Sec. V-C (Experiment Results): poor baseline results attributed to compounding errors and non-Markovian behaviour; behaviour degrades towards the end of an episode and the robot can pause indefinitely. The only related test is the Sec. VI-A ablation that adds chunking to two baselines; the two causes are not separated.

**4. supported.** Figure data table read from the image fm_data.png: ACT 1, 21, 44, 42, 41 at k = 1, 10, 100, 200, 400. Sec. VI-A: averaged over the 4 settings (2 simulated tasks x scripted or human data); temporal ensembling disabled for this experiment. The x-axis marks k = 1 as fully closed loop and k = 400 as fully open loop. Numbering confirmed: the caption is Fig. 8 in the arXiv PDF (I found it in the extracted PDF text) and Fig. 7 in the HTML.

**5. supported.** Same table: BC-ConvMLP 1, 1, 24, 25, 20; VINN 0, 6, 23, 27.25, 37. ACT is highest at k = 10, 100, 200 and 400 (at k = 1 it ties with BC-ConvMLP at 1). Note: at k = 400, VINN (37) is close to ACT (41).

**6. supported.** Sec. VI-A: the slight dip at k = 200 and 400 (close to open loop) is attributed to lack of reactive behaviour and the difficulty of modelling long action sequences.

**7. supported.** Sec. IV-A, Algorithm 2, Fig. 5: queried at every time step; weighted average of all predictions for the current step with w_i = exp(-m*i), w_0 the oldest; smaller m means faster incorporation of new observations; no extra training cost, only extra inference computation.

**8. supported.** The paper never gives a value for m (Algorithm 2 only lists the weight m as an input; I searched the full PDF text). imitate_episodes.py, eval_bc: query_frequency = num_queries, set to 1 when temporal_agg is on; k = 0.01; exp_weights = exp(-k * arange(n)), normalised. Extra detail: with this ordering the oldest prediction gets the largest weight.

**9. supported.** Figure table: no TE / with TE: ACT 44 / 47.3, BC-ConvMLP 25 / 29, VINN 37 / 17; bars labelled +3.3%, +4%, -20%. Sec. VI-A: highest success over k, same 4 settings, with and without TE tuned separately. The text itself gives only +4 and +3.3 and says VINN's performance dropped; the -20 is from the figure.

**10. supported.** Sec. VI-A: the authors hypothesise TE mostly helps parametric methods by smoothing modelling errors, while VINN retrieves ground-truth actions and does not suffer from this.

**11. corrected.** Everything is supported except the source of "32-dimensional". The paper's text never states the size of z (it only says z is projected to 512); Fig. 11 (PDF) is a vector diagram whose labels I could not decode, so I could not check it there (unchecked). The value 32 is in the official code (detr_vae.py: self.latent_dim = 32, marked with a TODO to tune) and in LeRobot (latent_dim = 32). Supported parts: [CLS] + joints + k-step action sequence into a BERT-like encoder, images left out for faster training (Sec. IV-B), mean and variance of a diagonal Gaussian, decoder conditions on images, joints and z, encoder discarded and z = 0 at test time so the policy is deterministic (Sec. IV-B, Fig. 4), beta = 10 (Table III). Small inconsistency worth knowing: Algorithm 1 writes the reconstruction loss as MSE, while the text (Sec. IV-C) and the code use L1.

**12. supported.** Figure table: scripted 59 with CVAE vs 58 without; human 35.3 vs 2 (Sec. VI-B text repeats 35.3% to 2%). The variant without the CVAE predicts the action sequence from the current observation and is trained with L1 loss only.

**13. supported.** Sec. III: four Logitech C922x webcams at 480x640, two on the follower wrists, one front, one top; teleoperation and recording at 50 Hz. Sec. IV: actions are the leader robots' joint positions; observations are the follower robots' current joint positions plus 4 images. Sec. IV-C: 7+7 = 14 joint values; action is a 14-dimensional vector of absolute joint positions.

**14. supported.** Sec. IV-C, no numbers given: L1 instead of L2 gave more precise modelling of the action sequence; delta joint positions as actions degraded performance. (See the MSE note under claim 11.)

**15. corrected.** All numbers are right: 15x20x512 feature map and 300x512 tokens per camera (Sec. IV-C); Table III: 4 encoder layers, 7 decoder layers, hidden 512, feed-forward 3200, 8 heads, lr 1e-5, batch 8, chunk 100, beta 10, dropout 0.1; about 80M parameters, trained from scratch for each task (Sec. IV-C). Correction: there is not one ResNet18 per camera in the official code. detr_vae.py builds one backbone (a single build_backbone call) and applies backbones[0] to every camera, with a HARDCODED comment; the paper only speaks of ResNet18 backbones in the plural. LeRobot also shares one backbone across cameras. Also, the paper's statement that the model is trained from scratch sits next to code that loads ImageNet weights for the ResNet18 (see claim 23).

**16. supported.** Sec. IV-C: about 5 hours on a single 11G RTX 2080 Ti; inference about 0.01 s on the same machine.

**17. supported.** Sec. V-B: 50 demonstrations per real task, 100 for Thread Velcro; 8 to 14 s or 400 to 700 steps at 50 Hz; about 10 to 20 minutes of data per task, 30 to 60 minutes wall-clock; 50 successful demonstrations for each simulated task, scripted or human. (Calculated: 50 x 8 to 14 s is 6.7 to 11.7 min, so "10 to 20 min" is the authors' rounding.)

**18. supported.** Table I and II, Sec. V-A and V-C: real tasks 1 seed, 25 evaluations; objects placed randomly along a 15 cm white reference line. Final success: Slide Ziploc 88, Slot Battery 96, Open Cup 84, Thread Velcro 20, Prep Tape 64, Put On Shoe 92. Baselines: BC-ConvMLP, BeT, RT-1, VINN all 0 final on Slide Ziploc and Slot Battery; only BeT was run on the Table II tasks, 0 final on all four. (Side note: the Table II caption speaks of the remaining three tasks but lists four.)

**19. supported.** Table I, 3 seeds x 50 evaluations: ACT Cube Transfer 86 (scripted) / 50 (human), Bimanual Insertion 32 / 20; best baseline BeT 27 / 1 and 3 / 0.

**20. supported.** Sec. VI-C and Fig. 8d: 6 participants; zip tie 33 s at 5 Hz vs 20 s at 50 Hz, cups 16 s vs 10 s; 62% longer teleoperation time; repeated measures design, p < 0.001. Human teleoperation, not the learned policy.

**21. supported.** detr_vae.py DETRVAE.forward: with a backbone, qpos always goes through input_proj_robot_state into the transformer; the no-backbone branch also uses qpos (plus env_state). No image-only path.

**22. supported.** transformer.py: build_transformer sets return_intermediate_dec = True, the decoder returns the stacked outputs of all layers, Transformer.forward returns that stack; DETRVAE takes element [0], which is the first decoder layer. imitate_episodes.py sets dec_layers = 7. LeRobot ACTConfig sets n_decoder_layers = 1 with a comment calling this a bug in the original code and linking tonyzhaozh/act issue 25. That issue (opened April 2024) is still open with no maintainer reply; one commenter reports lower training loss when using the last layer instead.

**23. supported.** backbone.py: torchvision ResNet built with pretrained = is_main_process() and norm_layer = FrozenBatchNorm2d. utils.py: EpisodicDataset length = number of episodes; each item draws start_ts = np.random.choice(episode_len); so one epoch = one random time step per episode. Extra detail: for real (non-simulated) data the action sequence starts one step earlier (start_ts - 1).

**24. supported.** README: simulated example uses --num_epochs 2000 (with kl_weight 10, chunk_size 100, hidden 512, batch 8, dim_feedforward 3200, lr 1e-5); for real data at least 5000 epochs or 3 to 4 times the length after the loss plateaus; success and smoothness can improve long after the plateau. Advice, not a measurement. Missed detail: the README also says a jerky policy, or one that pauses mid-episode, should simply be trained longer.

**25. supported.** configuration_act.py docstring: needs at least one image key or the environment state; may optionally work without observation.state. validate_features raises an error only if neither images nor environment state is present. modeling_act.py: the state projections are built only if robot_state_feature exists; without state the VAE encoder input is [CLS] plus the action embeddings.

**26. supported.** ACTConfig: chunk_size 100, n_action_steps 100, temporal_ensemble_coeff None (comment: ACT used 0.01), latent_dim 32, kl_weight 10.0, dim_model 512, vision_backbone resnet18 (ImageNet weights by default). __post_init__ refuses temporal ensembling unless n_action_steps is 1.

**27. supported.** State-free Table V: ACT with state 0 / 0.084, without 0.933 / 0.517; Diffusion Policy 0 / 0 vs 0.867 / 0.533; pi0 0 / 0 vs 0.984 / 0.584 (height / horizontal generalisation). Pick Pen with relative end-effector actions and dual wide-angle wrist cameras. Nuance: the default set-up seems to include the overhead camera too (Table IV, dual wide-angle with overhead camera, 0.983 / 0.583, matches the 0.984 / 0.584 of Tables III and V); only pi0 was used for most experiments; pick-and-place tasks had 300 episodes (Sec. IV-A).

**28. supported.** Table III (pi0, no state): relative EEF 1.0 / 0.984 / 0.584; absolute EEF, relative joint-angle and absolute joint-angle each 1.0 in domain and 0 / 0 for height and horizontal.

**29. supported.** Table IV: overhead only 0.217 / 0.133; single normal wrist + overhead 0.867 / 0.267; dual wide-angle wrist without overhead 1.0 / 1.0. Other rows the researcher left out: dual normal + overhead 0.920 / 0.400; single wide-angle + overhead 0.917 / 0.500; dual wide-angle + overhead 0.983 / 0.583.

**30. supported.** Abstract and Sec. I: state input leads to overfitting by memorising training trajectories; the "shortcut" wording is in Sec. III-B. Sec. VIII (Limitation): changing the background (for example relocating robot and table) may need fine-tuning.

**31. corrected.** Sec. 5.3 and Fig. 5 left support the trade-off and the finding that 8 steps was best for most tasks tested. But "(at 10 Hz)" is not in the paper for this result: the ablation is on simulated tasks (legend PushT and Square), whose control rate the paper does not state; 10 Hz is the rate of the real-world stations (Sec. 6.1, App. D). The tested horizons can be read from the figure's axis labels in the PDF text: 1, 2, 4, 8, 16, 32, 64, 128 steps. Also, the real pushing task (real Push-T) used an action horizon of 6, not 8 (Table 7).

**32. supported.** Sec. 4.3: single-step policies can overfit to pauses; BC-RNN and IBC often get stuck in real-world experiments when idle actions are not removed. Sec. 6.1 adds: LSTM-GMM got stuck near the T block in 8 of 20 runs, IBC left the T block too early in 6 of 20; the authors name not removing idle actions as one contributing cause, next to multimodality at stage transitions. No controlled test.

**33. supported.** Sec. 5.3: with simulated latency, Diffusion Policy kept peak performance up to 4 steps of latency (Fig. 5 right); velocity control more affected, likely through compounding errors.

**34. supported.** Sec. 6.1, Table 6, Table 7, App. D.0.1: UR5, policy at 10 Hz interpolated to 125 Hz; 2 of 5 cameras used, downsampled to 320x240; 136 demonstrations (Table 3); To 2, Ta 6, Tp 16. Table 6: Diffusion Policy (end-to-end CNN) IoU 0.80, success 0.95, duration 22.9 s; human 0.84, 1.00, 20.3 s; IBC 0.00 and 0.00, LSTM-GMM 0.20 and 0.10. Other Diffusion Policy variants were lower (transformer 0.65, ImageNet encoder 0.15, R3M 0.80).

**35. supported.** VQ-BeT Sec. 4.7 and Table 11 (App. B): closed loop (n = 1) 47/60 over six tasks (10 per task); receding horizon (n = 3) 0/30 (5 per task); attributed to imprecise motion control pushing the policy out of distribution. Caveats: the six tasks were chosen because Diffusion Policy did well on them in closed loop, and the open-loop arm had half the trials.

**36. supported.** Sec. 4.6: chunking was tried where possible and performance was lacking; small, fast models make chunking unnecessary. The ablation section covers conditional Kitchen, unconditional Ant and nuScenes; the chunking sentence itself does not name environments. Fig. 5 values: unchecked (the figure is not in the HTML version).

**37. corrected.** Table 3 numbers are right: with / without chunking LIBERO-90 0.90 / 0.76, Meta-World 0.79 / 0.78, DMC 0.70 / 0.74; Sec. 4.4 says chunking helps manipulation and mildly hinders locomotion. Correction: the chunk was 10 actions only for LIBERO-90 and Meta-World; for DeepMind Control it was 3 actions (Sec. 4.4, Table 4). Temporal ensembling is part of the method (Sec. 3.3, App. A.3), not named in Table 3.

**38. supported.** BID Sec. 3: Proposition 1 and Corollaries 2 (consistency, near-deterministic settings with an accurate model) and 3 (reactivity when forward inference is hard, for example stochastic settings); the paper states there is no universally optimal horizon.

**39. supported.** Sec. 5.1, Table 3: 0.0: 4.03 / 2.07 / 1.54 / 1.06 / 0.00; 0.2: 1.43 / 0.94 / 0.39 / 0.71 / 1.32; 0.4: 0.36 / 0.42 / 0.59 / 0.835 / 1.11 for horizons 1 / 3 / 5 / 7 / 10. Small wording point: the expert stays at s5 until it has been there for five steps; each forward move succeeds with probability 1 - delta.

**40. supported.** Table 1: vanilla open loop 64.0 / 26.9 / 13.0; vanilla closed loop 48.9 / 38.3 / 29.5; BID closed loop 54.4 / 45.3 / 31.7 (also BID open loop 66.1 / 31.4 / 16.0). Noise is temporally correlated Gaussian noise on executed actions, scaled by action size (Sec. 5.2); the base policy is the public LeRobot VQ-BeT checkpoint.

**41. supported.** App. A.5, Table 4, Average column: EMA open loop 34.9, closed loop 35.6; BID 37.8 and 43.8. EMA is described as temporal ensembling with lambda 0.5 (Table 6). Table 2 (A5000): 1 sample 13.2 ms; 8, 16, 32 samples 25.2, 25.9, 26.8 ms.

**42. supported.** Sec. 3 (Assumption 3.1, Proposition 3.1, Theorem 1) and Sec. 4 (Practice 2, Theorem 2). Wording: the assumption asks for open-loop incremental stability and Lipschitz policies (not "smooth"); the needed chunk length grows logarithmically in the stability and Lipschitz constants, not in task length; the example of open-loop stability is end-effector control with a low-level tracking controller.

**43. supported.** Sec. 5: tool_hang with full-state observations, executing slightly longer chunks improved success drastically while the prediction horizon had only a transient effect (Fig. 5); naive chunking disastrous on HalfCheetah (Fig. 10 right); noisy labels catastrophic (Fig. 10 left). Plotted values: unchecked.

**44. supported.** RTC Sec. 4.2 and Fig. 6: synchronous inference degrades linearly with added delay; RTC best at all delays; faster than synchronous even with pauses removed. Set-up: pi0.5, H = 50 at 50 Hz, synchronous executes 25 then pauses, six bimanual tasks, 10 trials each (480 episodes). Exact throughputs only in the figure: unchecked.

**45. supported.** Sec. 4.1: TE poor across the board, also at zero delay, explained by multimodality (averages of valid actions are not necessarily valid). Sec. 4.2: TE could not run at +100 and +200 ms because oscillations triggered the protective stop.

**46. supported.** Abstract and Sec. 2: synchronous pauses slow execution and change the robot's dynamics, causing a train-test distribution shift. Argument.

**47. supported.** Tables I and II, Sec. V: average LIBERO success 76.5 to 90.2 with parallel decoding plus chunking; throughput 4.2 to 108.8 actions per second on an A100 with K = 8; latency 0.0629 s (parallel decoding only) to 0.0735 s (with chunking); full chunks executed before replanning.

**48. supported.** SAIL Table 1 (DP vs DP-Fast success): Can 0.97 / 0.87, Square 0.83 / 0.55, Mug 0.68 / 0.56. Table K.7 (ACT vs SAIL-ACT): Lift 1.00 / 0.94, Can 0.77 / 0.56, Square 0.51 / 0.43, Stack 0.73 / 0.68; SAIL-ACT speed-up over demonstration 3.71, 2.72, 2.34, 1.72 (so 1.7 to 3.7). SAIL-ACT uses temporal ensembling. Throughput-with-regret still rose with SAIL (Lift 0.40 to 1.50).

**49. corrected.** App. B, Eq. 2: TPR = (1/N) x sum over all N rollouts of [S_i / t_i - (1 - S_i) / t_max]. So the average is over all rollouts, not only successful ones: each success adds 1/duration, each failure subtracts 1/t_max, then divide by N.

## 3. Challenge to the conclusion

The evidence summary is mostly accurate, with three fixes:
- "Diffusion Policy found about 8 steps at 10 Hz best": the 8-step result is from simulation at an unstated rate; the real pushing task used 6 steps at 10 Hz.
- "A 2025 study ran ACT without state, generalising better with relative actions" joins two different results. The action-type comparison (relative vs absolute, Table III) was done with pi0. For ACT the study only compared with and without state (Table V), with relative end-effector actions in both rows.
- Image-only input is not a choice we earned from evidence: the course already requires the one camera as the only input. The state-free paper is reassurance, not a reason.

Does the evidence support the judgement for our situation? Only partly.

1. **What the ACT ablation measured.** The 1% to 44% jump is simulation at 50 Hz, joint-position targets, several cameras plus joint state, tested on the same tasks it was trained on, averaged over scripted and human data. At k = 1 even BC-ConvMLP scored 1%, so the low k = 1 figure says as much about the set-up as about chunking.
2. **Our horizon is shorter, but not short.** Calculated: 18,500 pictures / 208 demonstrations is about 89 decisions per track. ACT's real episodes have 400 to 700 steps; at k = 100 without ensembling that is only 4 to 7 decisions per episode. Our 89 decisions sits between ACT's worst (k = 1) and best (k = 100) cases, so compounding errors can still matter. "Gains will likely be smaller" is plausible but nothing measured it.
3. **Chunk size should be judged in distance, not step count.** Five to ten 1 cm steps is 5 to 10 cm of track. If the forearm camera sees less line than that ahead, a long chunk must guess the track beyond the picture. On the six training sheets the model can memorise shapes; on unseen tracks it cannot. None of the cited studies tests unseen layouts. ACT moved objects along a 15 cm line on the same tasks; the state-free study moved heights and object positions. So the evidence says nothing direct about the exam's unseen tracks.
4. **The "price of not looking" evidence is from extreme cases.** BID used artificial correlated noise; VQ-BeT's 0/30 was a low-cost mobile robot with imprecise motion, on tasks chosen because closed loop worked. The closest analogue to our task, real Push-T with a precise UR5 arm, ran 6 steps open loop and reached 95%. For us the arm replays steps precisely (not checked here); the open question is the puck slipping against the magnet block, which the theory paper (Zhang et al.) would call a lack of open-loop stability.
5. **Stop-and-go changes what chunking can buy.** There is no latency problem (the robot waits), so the latency and real-time-chunking results do not apply. Temporal ensembling needs a picture at every step, so it saves no time. The only time gain is skipping picture-and-decide for some steps. How much of the 1.2 s per step that is has not been measured. Illustrative calculation with a guessed 0.4 s for picture-and-decide: executing 3 steps per picture saves about 0.27 s per step (about 22%). With 0.1 s it would be about 6%. Measure this first. Our data were recorded stop-and-go with the robot at rest for each picture, so stop-and-go execution matches training. Chaining steps into one smooth move would add the train-test shift RTC describes, and SAIL shows that naive speed-ups cost success.
6. **Move names are classes.** None of the sources predicts chunks of discrete move classes. Averaging actions (temporal ensembling) is not defined for classes. Averaging class probabilities over overlapping chunks would be the analogue, but it is untested.
7. **The CPU constraint is not addressed.** ACT as published is about 80M parameters, 4 cameras at 480x640, and 5 GPU hours. LeRobot's own compute guide says not to train on CPU only. Its indicative figure for ACT on Apple MPS is 6 to 14 h for 5 passes over about 45,000 frames. In the ACT code an epoch draws one random time step per episode, so the README's 5000 epochs are not 5000 passes over our 18,500 pictures. The code's single shared backbone and the effectively one-layer decoder suggest a much smaller model is enough, which matters for a laptop.

What the evidence does support: train with a short chunk target (5 to 10 steps) and execute 1 step per picture by default (closed loop, like Diffusion Policy's closed loop on the Stretch and BID's closed loop under noise). Test 2 or 3 steps per picture only for speed, and only on held-out tracks. The proposed experiment design is sound. Make it bigger than one held-out sheet if CPU time allows (leave-one-sheet-out). Use enough runs per condition to tell differences apart: ACT used 25 trials per real task; at 10 runs only large gaps are meaningful. Rank by finishing first, time second. SAIL's throughput-with-regret (corrected formula above) does this in one number. Also look at the dashed loop and the crossing on their own: those are where the BID theory says consistency from longer chunks could help, and where gaps or crossings make one picture ambiguous.

## 4. Evidence the researcher missed (all checked unless marked)

1. Diffusion Policy real Push-T (Table 7, Table 6): the only real pushing task in the set. It executed 6 of 16 predicted steps open loop at 10 Hz on a precise arm and reached 95% with 136 demonstrations.
2. BID Table 4: under noise, temporal ensembling (EMA) in closed loop was worse than plain closed loop: 35.7 vs 38.3 at noise 1.0, 18.4 vs 29.5 at noise 1.5. It was better only without noise (52.6 vs 48.9). The averages in claim 41 hide this.
3. BAKU Table 3, observation history: adding history with a last-step loss collapsed performance (0.54 / 0.08 / 0.37 vs 0.90 / 0.79 / 0.70); with a multi-step loss it was about equal. This is relevant if history is considered instead of chunking for non-Markovian moments.
4. Zhang et al. Sec. 5: executing longer chunks mattered, while a longer prediction horizon had only a transient effect. So "predict 10, execute 1" may get little from the extra predicted steps.
5. ACT README: jerky or mid-episode pausing policies should be trained longer. Training length, not chunk size, is the authors' first fix for pauses.
6. ACT code facts that matter for a laptop: one ResNet18 shared by all cameras, with ImageNet weights and frozen batch norm; effectively one decoder layer (issue 25, LeRobot uses 1); one sample per episode per epoch.
7. LeRobot compute guide: CPU-only training not recommended. ACT on Apple MPS takes about 6 to 14 h for 5 passes over about 45,000 frames (indicative figures from the guide, not a measurement). Imitation learning typically converges in 5 to 10 passes over the data.
8. State-free Table IV rows the researcher left out: field of view mattered (single normal wrist + overhead 0.867 / 0.267; single wide-angle + overhead 0.917 / 0.500; dual wide-angle alone 1.0 / 1.0). With one camera, what the camera sees may matter more than chunk length for unseen tracks.
9. VQ-BeT Table 11 caveat: tasks were picked where closed loop worked; open loop had 5 trials per task vs 10.
10. LeRobot's default executes all 100 predicted actions before looking again (n_action_steps = 100). Anyone using LeRobot defaults would run open loop for 100 steps, which for us is about a whole track.
11. Not found in any source (gaps, not evidence): chunking with discrete move classes; chunking evaluated on unseen paths or layouts; chunking in a stop-and-go loop at about 1 cm per step; CPU training of ACT-like models.

## 5. What I could not check

- Fig. 11 of ACT (the detailed architecture diagram): its labels are in an encoded font I could not decode, so whether it shows z = 32 is unchecked.
- Plotted values in VQ-BeT Fig. 5, Zhang et al. Figs. 5 and 10, and RTC Fig. 6: not read (text statements checked only).
- Control rate of Diffusion Policy's simulated tasks: not stated in the paper.

Working files in this folder: tools/pdf_text_search.py (standard-library PDF text extractor used for the ACT and Diffusion Policy PDFs).
