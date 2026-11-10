# Pretrained vision backbones for control: literature notes

Key: `pretrained_representations`
Written: 9 November 2026, by a research sub-agent for Team M (Project 6, Niryo Ned2 puck pushing).
Status: reading notes for discussion. Nothing here has been tested on our data or our laptop.

How to read these notes

- **Measured** means a paper ran the experiment and reported the number.
- **Authors' view** means the authors' explanation or guess, not something they measured.
- **My judgement** means my own reading for our project. It is not evidence.
- **Calculation** means arithmetic I did from published model sizes, with the assumptions stated.
- Every paper listed in section 3 was opened (the URL given is the page I read). Section 9 lists papers I only saw cited.
- I read most papers through their arXiv HTML pages (arxiv.org/html or ar5iv). The final conference version can differ a little from the arXiv version.

---

## 1. The question

Our model sees one 640x480 picture from the forearm camera and must choose the next move (8 picture-frame directions, stop, small motion, plus the exact step). We have 208 scored demonstrations, about 18,500 pictures, on six sheets. The exam adds unseen tracks of the same size and line thickness. We train on a 6-core laptop CPU with no NVIDIA GPU.

Which image backbone should turn the picture into features: an ImageNet ResNet, a robot-specific model (R3M, MVP, VC-1), a self-supervised ViT (DINO, DINOv2) or CLIP? Should it be frozen or fine-tuned? Which one is the safest for unseen tracks with a small dataset and a CPU?

## 2. Short answer

What the evidence shows:

1. No pretrained backbone is best everywhere. The ranking changes with the benchmark and with the policy learning method (Majumdar et al. 2023; Hu et al. 2023; Dong et al. 2026).
2. Robot-specific backbones (R3M, MVP, VC-1) look best mainly in frozen, few-demonstration simulation tests. They were not more robust to lighting, texture or distractor changes than ImageNet-trained models (Burns et al. 2024), and ImageNet pretraining was as good as or better than egocentric human video in real-robot tests (Dasari et al. 2023).
3. With about 100 to 200 demonstrations, a ResNet-18 trained end-to-end with image-shift augmentation matched or beat frozen pretrained features (Hansen et al. 2023; Chi et al. 2023; Mandlekar et al. 2021).
4. Fine-tuning helps ResNets when it is done with augmentation and a small learning rate. Fully fine-tuning large ViTs on small data can collapse (VC-1: Adroit 59.3 to 15.9).
5. Among frozen models, DINO-trained ViTs were the most robust to appearance changes, and a ViT's emergent segmentation predicted that robustness. DINOv2 ViT-S, however, did worse than expected in the same study.
6. No study I found tested what our exam tests: new track geometry with the same look.

My judgement: the safest default is an ImageNet-pretrained ResNet-18 (11.7 million parameters, 1.81 GFLOPS per 224x224 image in torchvision's count), fine-tuned end-to-end with a lower backbone learning rate, frozen BatchNorm, and shift and colour augmentation, reading spatial feature maps instead of one pooled vector. Compare it once against frozen DINO ViT-S/16 (or DINOv2 ViT-S/14) patch tokens with a small attention-pooling head. Decide with a leave-one-sheet-out test, not with accuracy on the training sheets. Do not use ViT-L models (VC-1, MVP-L) on the laptop.

---

## 3. Evidence, paper by paper

### 3.1 Parisi, Rajeswaran, Purushwalkam, Gupta (2022). "The Unsurprising Effectiveness of Pre-Trained Vision Models for Control". ICML 2022, PMLR 162:17359-17371.

Opened: https://arxiv.org/abs/2203.03580 and https://ar5iv.labs.arxiv.org/html/2203.03580

Set-up (checked): frozen encoders, behaviour cloning (BC). Habitat navigation with 10,000 trajectories per scene; MuJoCo control (Adroit, DeepMind Control, Franka Kitchen) with 25 to 100 demonstrations per task (Sec. 3.3, App. A.2). Models: supervised ImageNet ResNet-34/50, MoCo-v2 ResNet-50, CLIP ResNet-50 and ViT-B/32, random frozen features, and a small convolutional network trained from scratch (Sec. 3.2, Fig. 6). Images 256x256, last three frames (appendix). Fine-tuning was left to future work (Sec. 5). Simulation only.

Measured (checked):
- Every pretrained model clearly beat random features and the from-scratch encoder (Sec. 4.1, Fig. 6). The paper does not mention augmentation for the from-scratch baseline (this absence is unchecked; see Hansen et al. below, who argue it matters).
- No single model was best in all four domains. Self-supervised MoCo was better on average than supervised ImageNet and CLIP (Sec. 4.1, Fig. 6). No off-the-shelf model matched ground-truth state in MuJoCo (Fig. 6).
- Features from early layers (layer 3) worked better for MuJoCo control; the last layer worked better for navigation (Sec. 4.4, Fig. 9). A MoCo model using layers 3, 4 and 5 together matched or beat ground-truth state in all domains (Sec. 4.5, Fig. 10).
- Crop-only augmentation during pretraining was as good as or better than the other choices on average (Sec. 4.3, Fig. 8).
- Pretraining on in-domain data was not much better than ImageNet, and for MuJoCo it was worse (Sec. 4.2, Fig. 7).

One quote: "pre-trained visual representations can be competitive or even better than ground-truth state representations" (abstract).

Authors' view: crop invariance may emphasise the relative position of agent and objects (Sec. 4.3); navigation needs semantic cues held in later layers (Sec. 4.4).

My judgement: our task is fine control from low-level geometry (a black line and a red puck), so the early-layer result is the most relevant one: do not throw away spatial detail by using only the final pooled vector.

### 3.2 Nair, Rajeswaran, Kumar, Finn, Gupta (2022). "R3M: A Universal Visual Representation for Robot Manipulation". CoRL 2022, PMLR 205:892-909 (published 2023).

Opened: https://ar5iv.labs.arxiv.org/html/2203.12601 and https://proceedings.mlr.press/v205/nair23a.html

Set-up (checked): ResNet-50 pretrained on Ego4D human video with time-contrastive learning, video-language alignment and a sparsity penalty (Sec. 3.3, App. A.2). Frozen encoder plus a 2-layer MLP trained by BC (Sec. 4.1, App. B.5). 5, 10 or 25 demonstrations (Meta-World, Franka Kitchen), 25 to 100 (Adroit).

Measured (checked):
- Across 12 simulated tasks R3M averaged about 62.4% success, more than 20 points above training from scratch and more than 10 points above the best baselines (CLIP, MoCo) (Sec. 4.3, Table 1).
- On a real Franka with 20 demonstrations per task, R3M averaged 56% against CLIP's 24% over five tasks (Table 3).
- Removing the language-alignment loss hurt the most (Sec. 4.4).

My judgement: strong evidence for R3M in the frozen, very-few-demonstration regime. Our regime (about 200 demonstrations, possible fine-tuning) is different, and later studies (3.7, 3.9, 3.10, 3.13, 3.14) found R3M weaker under distribution shift or with fine-tuning.

### 3.3 Xiao, Radosavovic, Darrell, Malik (2022). "Masked Visual Pre-training for Motor Control". arXiv:2203.06173.

Opened: https://arxiv.org/abs/2203.06173 (abstract only)

Measured (checked, abstract): a frozen masked-autoencoder (MAE) encoder with reinforcement learning on top beat supervised encoders by up to 80% absolute success in simulation. RL, not imitation; I read only the abstract.

### 3.4 Radosavovic, Xiao, James, Abbeel, Malik, Darrell (2022). "Real-World Robot Learning with Masked Visual Pre-training". CoRL 2022, PMLR 205:416-426 (published 2023).

Opened: https://ar5iv.labs.arxiv.org/html/2210.03109 and https://proceedings.mlr.press/v205/radosavovic23a.html

Set-up (checked): MAE-pretrained ViT-S (22M), ViT-B (86M) and ViT-L (307M) on 4.5 million images from Ego4D, ImageNet and hand-object video (Sec. 4). Frozen encoder (Sec. 3.2). BC with 80 demonstrations per task, wrist camera, xArm7, up to 5 Hz (Sec. 4).

Measured (checked):
- MVP was up to 75% better than CLIP and up to 81% better than supervised ImageNet and training from scratch (abstract, Sec. 1, Fig. 4).
- MVP needed about half the demonstrations to match CLIP trained on 80 (Sec. 5.1, Fig. 5).
- PickFruit: R3M 31.3%, CLIP 18.8%, MVP ViT-S 68.8%, ViT-B 93.8%, ViT-L 100% (Table 1).
- Fine-tuning ViT-S (HoI) and CLIP ResNet-50 gave only marginal gains (Sec. 5.5).
- ViT-L costs about 64 GFLOPs per 224x224 image, roughly 15 times ResNet-50 (Sec. 3.1).

Authors' view: freezing avoids overfitting large encoders and saves memory and time (Sec. 3.2).

### 3.5 Majumdar et al. (2023). "Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?" (VC-1). NeurIPS 2023.

Opened: https://ar5iv.labs.arxiv.org/html/2303.18240 and https://mlanthology.org/neurips/2023/majumdar2023neurips-we

Set-up (checked): CortexBench, 17 tasks. Imitation learning for Adroit (100 demonstrations), Meta-World (25), DeepMind Control (100), TriFinger (100) (App. A.2). Frozen encoders in Sec. 4 and 5; adaptation in Sec. 6. VC-1 is a ViT-L trained with MAE on 5.6 million frames (Ego4D plus other egocentric, navigation and ImageNet images) (Tables 3, 6).

Measured (checked):
- No existing pretrained model was best on all benchmarks: R3M led Adroit, Meta-World and DeepMind Control; MVP ViT-L led TriFinger, ImageNav and Mobile-Pick; CLIP led ObjectNav (Sec. 4, Table 2).
- Mean success: VC-1 68.7, MVP ViT-L 67.5, R3M 58.0, CLIP 57.0 (Table 4). VC-1 was best on average but not everywhere.
- Full end-to-end fine-tuning of VC-1 collapsed in the few-demonstration tasks: Adroit 59.3 to 15.9, Meta-World 88.8 to 22.7, DeepMind Control 66.9 to 6.7. It helped in the large-data navigation tasks. MAE adaptation (continuing MAE pretraining on task images) helped the few-demonstration tasks: 72.0, 96.0, 80.9 (Sec. 6, Table 5).
- Real Franka, 4 tasks, 10 trials each: MVP 55%, VC-1 frozen 70%, VC-1 with MAE adaptation 85% (Sec. 7).
- Training VC-1 and its variants took over 10,000 GPU-hours (Introduction).

Authors' view: the fine-tuning collapse comes from overfitting a 307M-parameter model on at most about 50,000 frames (Sec. 6).

My judgement: our 18,500 pictures are well inside the regime where full fine-tuning of a big ViT failed.

### 3.6 Hansen, Yuan, Ze, Mu, Rajeswaran, Su, Xu, Wang (2023). "On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline". ICML 2023, PMLR 202:12511-12526.

Opened: https://ar5iv.labs.arxiv.org/html/2212.05749, https://arxiv.org/html/2212.05749v2 and https://proceedings.mlr.press/v202/hansen23c.html

Set-up (checked): learning from scratch (LfS) with a shallow ConvNet and random-shift augmentation, against frozen PVR (MoCo-v2 ResNet-50), MVP (ViT-S) and R3M (ResNet-50) (Sec. 2, Table 1). BC with 100 demonstrations (Adroit, DeepMind Control) and 10 or 20 on a real xArm 7 with a third-person camera; also RL.

Measured (checked):
- LfS with augmentation was competitive overall. Examples (Table 2, LfS no aug / LfS with aug / best pretrained): Adroit Pen 22.0 / 74.8 / 81.3; Relocate 16.9 / 51.4 / 47.5; real Pick 25 / 55 / 35; real Reach 80 / 85 / 90.
- With 10 demonstrations the frozen models did marginally better; with 100, LfS was generally better (Fig. 8, averaged over DeepMind Control tasks).
- Fine-tuning helped only with augmentation (Table 5, Adroit): R3M 64.0 frozen, 55.5 fine-tuned, 80.5 fine-tuned with augmentation; PVR 52.9, 50.5, 65.0; MVP (ViT) 44.0, 18.7, 31.1; LfS 19.4 without and 63.1 with augmentation. The authors note ResNets were easier to fine-tune than the ViT.
- Under visual shifts (DeepMind Control Generalisation Benchmark), LfS with colour jitter scored 53.1 against 37.2 for the best frozen model on random colours, and LfS with random overlay 46.6 against 9.2 on video backgrounds (Table 3).
- Frozen features made BC training at least an order of magnitude faster because features are cached once, but the small LfS network was faster at inference (Sec. on compute, Table 4).

Authors' view: the weak showing of pretrained models comes from a domain gap and from visually simple benchmarks; pretraining may matter more on visually complex, multi-task problems (conjecture).

My judgement: this is the closest evidence to our setting (simple scenes, about 100 to 200 demonstrations). It says augmentation matters at least as much as pretraining.

### 3.7 Hu, Wang, Li, Gao (2023). "For Pre-Trained Vision Models in Motor Control, Not All Policy Learning Methods are Created Equal". ICML 2023, PMLR 202:13628-13651.

Opened: https://ar5iv.labs.arxiv.org/html/2304.04591 and https://proceedings.mlr.press/v202/hu23h.html

Set-up (checked): 14 frozen models (9 ResNet-50, 5 ViT-B/16, including DINO, MAE, CLIP, R3M, VIP, MoCo) on 21 tasks; BC with 1 to 25 demonstrations (Meta-World, Franka Kitchen) or 5 to 50 (Robosuite) (Sec. 3, App. C).

Measured (checked):
- How useful a pretrained model is depends heavily on the policy learning method; RL evaluation was very variable (abstract, Sec. 4.1).
- Under BC, R3M was best overall, VICRegL (ImageNet-only) close behind; ResNet models generally beat ViT models; no model was best in all environments (Sec. 4.2, Fig. 7).
- A linear probe that predicts the true environment state from frozen features predicted BC success (Franka Kitchen: Pearson r = -0.78 between probe loss and success) (Sec. 4.4, Fig. 9).

My judgement: a cheap probe (for example, predict puck position or line direction from frozen features) could rank backbones before full training.

### 3.8 Dasari, Srirama, Jain, Gupta (2023). "An Unbiased Look at Datasets for Visuo-Motor Pre-Training". CoRL 2023, PMLR 229:1183-1198.

Opened: https://arxiv.org/abs/2310.09289, https://arxiv.org/html/2310.09289 and https://proceedings.mlr.press/v229/dasari23a.html

Set-up (checked): MAE ViT (ViT-B size) pretrained on 1 million images from each dataset (Sec. 3, 4.1, App. A). The whole policy, encoder included, was fine-tuned end-to-end by BC (Sec. 3). Real Franka, third-person camera, 50 teleoperated demonstrations per task, three tasks (App. C).

Measured (checked, Table 1, real-robot average success with standard errors of about 7 points):
- ImageNet 41%, Kinetics 44%, 100 Days of Hands 38%, RoboNet 30%, Ego4D 28%, VC-1 33%, MVP 25%, no pretraining 0% (0% on all three real tasks; 38% average in simulation).
- Doubling the pretraining data (Soup-1M to Soup-2M) did not help (41% and 41%); adding 1 million hand-object frames reached 60% (Table 2).
- Simulation results barely predicted real results (R squared 0.32) (Fig. 5).
- Dropout (p = 0.2) in the policy almost consistently improved real-world success (Sec. 5, Fig. 4); a disturbed block-stacking task went from 10% to 24%.
- A ResNet-18 SimCLR repeat (App. D): ImageNet 32%, Ego4D 25%, 100 Days of Hands 44%, R3M 8% average real success.

My judgement: for real robots, ImageNet is a good pretraining source; robot-specific datasets gave no clear advantage. The 0% for no pretraining applies to a ViT, which needs pretraining; it does not contradict Hansen's small ConvNet result.

### 3.8b Silwal et al. (2023, revised 2024). "What do we learn from a large-scale study of pre-trained visual representations in sim and real environments?" arXiv:2310.02219 (listed as ICRA 2024 by search results; venue page not opened).

Opened: https://arxiv.org/abs/2310.02219 and https://arxiv.org/html/2310.02219v2

Set-up (checked): R3M, CLIP, MVP, VC-1 Base and VC-1 Large; BC on a TriFinger (30 demonstrations) and a Franka (30 demonstrations for reach and bottle pick-up, 150 for drawer opening); image-goal navigation on a Stretch robot trained in simulation (Sec. III).

Measured (checked):
- After aligning the simulated and real set-ups, simulation rankings predicted real rankings (Spearman 0.895 in Sec. IV-B; the introduction gives 0.929, an inconsistency in the paper). Rankings from the standard CortexBench simulation did not (Spearman -0.475) (Sec. IV-B).
- Fine-tuning plus augmentation gave small real-world gains: VC-1 Base 59% against 57% frozen without augmentation, VC-1 Large 47% against 45% (Sec. IV-D, Table IV).
- No ImageNet-only ResNet baseline was tested.

My judgement: this disagrees with Dasari et al. on whether simulation predicts reality, but agrees that fine-tuning large ViTs brings little on small real datasets.

### 3.9 Burns, Witzel, Hamid, Yu, Finn, Hausman (2023, published 2025). "What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?" CoRL 2024, PMLR 270:4525-4545.

Opened: https://arxiv.org/abs/2312.12444, https://arxiv.org/html/2312.12444 (v1) and https://proceedings.mlr.press/v270/burns25a.html (I read v1; the CoRL version may differ.)

Set-up (checked): 15 frozen models, among them supervised ImageNet ResNet-50 and ViT-S, DINO ResNet-50 and ViT-S/16, MoCo-v3, MAE, R3M, MVP, VIP, CLIP ViT-B/16 and DINOv2 ViT-S/14 (Table 2, App. A.1). BC with a 2-layer MLP and proprioception, 10 demonstrations per task, five tasks each in Franka Kitchen and Meta-World (Sec. 3, App. A.4). Shifts: lighting, textures, distractor objects.

Measured (checked):
- Models designed for manipulation did not generalise better out of distribution; the best of them ranked seventh of 15 (Fig. 4).
- ViT-DINO was the strongest under shifts, beating all three manipulation models on every shift except Meta-World distractors (Sec. 4). Six of the top seven models under shift were ViTs (Sec. 4).
- Among ViTs, the segmentation score (Jaccard index of the last-layer CLS attention map against PASCAL VOC masks) was strongly correlated with success under shift; ImageNet accuracy was a poor predictor; shape bias showed a correlation only for ResNets (Sec. 5.1, Fig. 5, App. A.8). The text gives no correlation numbers.
- DINOv2 ViT-S/14 did worse than expected both in and out of distribution, without explanation (App. A.1).
- On the real ALOHA robot (50 demonstrations, one task, 10 rollouts under a placement and lighting change), MoCo-v3 reached 40% and MVP 0% (Table 1, Sec. 5.4).
- With fine-tuning and 25 demonstrations in Franka Kitchen, models not designed for manipulation still did better (App. A.10, Fig. 12).

One quote: "emergent segmentation ability is a strong predictor of out-of-distribution generalization among ViT models" (abstract).

Authors' view: the choice of augmentations during pretraining matters more than whether labels were used (Sec. 4).

My judgement: the shifts tested are about appearance. Our exam shift is mainly geometry (new track shapes), with appearance as a secondary risk (exam room light, new table since 6 November). The black line and red puck are high-contrast, so segmentation is probably easy for every backbone; this metric matters most if lighting or background changes.

### 3.10 Chi, Feng, Du, Xu, Cousineau, Burchfiel, Song (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion". RSS 2023.

Opened: https://arxiv.org/html/2303.04137 (extended arXiv version), https://www.roboticsproceedings.org/rss19/p026.html, and the released config https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/config/train_diffusion_unet_real_pretrained_workspace.yaml

Set-up (checked): the default encoder is a ResNet-18 without pretraining, with spatial-softmax pooling and GroupNorm instead of BatchNorm, trained end-to-end (Sec. 3.2).

Measured (checked):
- Encoder ablation on robomimic Square with proficient-human demonstrations (Sec. 5.4, Table 5; this dataset has 200 demonstrations according to Mandlekar et al., 3.11). Success as from scratch / frozen pretrained / fine-tuned: ResNet-18 (ImageNet-21k) 0.94 / 0.58 / 0.92; ResNet-34 0.92 / 0.40 / 0.94; ViT-B/16 (CLIP) 0.22 / 0.70 / 0.98.
- Fine-tuning with a learning rate 10 times smaller than the policy network's gave the best results (Sec. 5.4).
- Real Push-T (push a T-shaped block to a target, 136 demonstrations, two cameras at 320x240): end-to-end 95% success, R3M 80%, ImageNet 15% (Sec. 6.1, Table 6). The R3M version was jittery and got stuck more often (Sec. 6.1).
- The paper does not say whether the R3M and ImageNet encoders were frozen in Push-T. The released config for real pretrained runs sets `freeze_encoder: True` with an ImageNet ResNet-18 at 224x224 and no random crop. That they were frozen is my inference (unchecked).

Authors' view: frozen pretrained features are not what the policy needs; end-to-end training remains the most effective way to use images (Sec. 5.4, 6.1).

My judgement: Push-T is the published task closest to ours (planar pushing with a rod, about 140 demonstrations). It favours end-to-end training of a small ResNet.

### 3.11 Mandlekar et al. (2021, published 2022). "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation" (robomimic). CoRL 2021, PMLR 164:1678-1690.

Opened: https://ar5iv.labs.arxiv.org/html/2108.03298 and https://proceedings.mlr.press/v164/mandlekar22a.html

Measured (checked): image policies used a ResNet-18 followed by spatial softmax (App. D.1); the proficient-human datasets have 200 demonstrations (Sec. 3.2). Image-based BC-RNN nearly matched policies given the true state (Sec. 4.3, Table 3). Removing pixel-shift augmentation cut success by 47% (relative) on Square and 35% on Transport (Sec. 4.3, Fig. 2a).

### 3.12 Zhao, Kumar, Levine, Finn (2023). "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT). RSS 2023.

Opened: https://ar5iv.labs.arxiv.org/html/2304.13705, https://www.roboticsproceedings.org/rss19/p016.html and the code https://raw.githubusercontent.com/tonyzhaozh/act/main/detr/models/backbone.py

Checked: ResNet-18 backbones on 480x640 images (Sec. IV-C), 50 demonstrations per task (Sec. V-B), about 5 hours of training on one RTX 2080 Ti and about 0.01 s inference (Sec. IV-C). The paper text does not say whether the backbone starts from ImageNet. In the code, the backbone is built with torchvision's `pretrained=` option and frozen BatchNorm statistics, and the convolution weights stay trainable. That torchvision's pretrained ResNet weights are ImageNet weights is my knowledge of torchvision (the model card I opened describes the newer `IMAGENET1K_V1` weights option), so this detail is unchecked.

My judgement: a widely copied recipe for about 50 demonstrations: ImageNet ResNet-18, frozen BatchNorm, fine-tuned end-to-end.

### 3.13 Xie, Lee, Xiao, Finn (2023). "Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation". arXiv:2307.03659 (listed as ICRA 2024 by dblp; venue page not opened).

Opened: https://ar5iv.labs.arxiv.org/html/2307.03659

Measured (checked):
- In simulation, frozen R3M and CLIP (ResNet-50) fitted the training environments but left a large generalisation gap; R3M generalised poorly to most factors except new distractors; CLIP was similar to a CNN trained from scratch except on new object textures (Sec. 4.2, 5.2, Fig. 7, Fig. 11).
- On the real robot, new camera positions were hardest (45.8% against 91.7% without shift), then table texture (52.8%); random crops helped most for camera position and table texture (Sec. 5.1 to 5.2, Fig. 1, Fig. 6).
- More training environments shrank the gap (5 to 100 environments: from 0.4 to under 0.1) (Fig. 5a).

My judgement: more variety in the training data (our six sheets, both directions) is likely to matter more than the backbone.

### 3.14 Lin, So, Mahalingam, Liu, Abbeel (2023). "SpawnNet: Learning Generalizable Visuomotor Skills from Pre-trained Networks". arXiv:2307.03567 (venue not confirmed).

Opened: https://arxiv.org/abs/2307.03567 and https://ar5iv.labs.arxiv.org/html/2307.03567

Measured (checked, real robot, about 90 to 102 demonstrations per task, unseen object instances):
- SpawnNet (frozen DINO ViT-S/8 features from several layers fused into a trainable convolutional stream): Place Bag 86.7%, Hang Hat 80.0%, Tidy Tools 88.6%; LfS with augmentation 66.7%, 60.0%, 33.3%; frozen R3M 20.0%, 20.0%, 15.2% (Table 6).
- Ways of using DINO on Place Bag: frozen 11.7%, full fine-tune 8.3%, last-layer fine-tune 15.0%, LoRA 28.3%, SpawnNet 86.7% (Table 3).

My judgement: a frozen ViT alone was weak in the real world, and so was full fine-tuning; mixing pretrained multi-layer features with a small trainable spatial network worked best. A ResNet fine-tuned with a small learning rate is a simpler relative of this idea.

### 3.15 Shang et al. (2024). "Theia: Distilling Diverse Vision Foundation Models for Robot Learning". CoRL 2024 (per arXiv comments).

Opened: https://arxiv.org/abs/2407.20179 and https://arxiv.org/html/2407.20179v2

Measured (checked): for transformer encoders, spatial (patch) tokens consistently beat the CLS token for robot learning, including for MVP and VC-1 (Sec. 4.2, Table 3c). In the real-world tests, frozen encoders reached 0% on one task (toy microwave cooking) where fine-tuning worked (App. D.3.1). Frozen CortexBench means: Theia-B 79.8, MVP-L (spatial) 77.4, R3M 76.5, VC-1-L (spatial) 69.6 (Table 1).

### 3.16 Tsagkas et al. (2025, revised 2026). "Attentive Feature Aggregation or: How Policies Learn to Stop Worrying about Robustness and Attend to Task-Relevant Visual Cues". arXiv:2511.10762 (preprint).

Opened: https://arxiv.org/html/2511.10762v2

Measured (checked): 14 frozen models, 10 Meta-World tasks, 25 demonstrations each. Replacing the CLS token or average pooling with a learned cross-attention pooling head raised in-distribution success from 63.1% to 66.4% and often improved success under visual perturbations a lot; spatial softmax gave a slight in-distribution gain but was worse under perturbation (Sec. V-B, Fig. 4). On a real planar-pushing task, success under perturbation went from 0% to 100% (10 in-distribution and 16 perturbed rollouts, a small test) (Table II).

### 3.17 Dong, Man, Tokmakov, Wang (2026). "Capturing Visual Environment Structure Correlates with Control Performance". arXiv:2602.04880 (preprint).

Opened: https://arxiv.org/html/2602.04880v1

Measured (checked): nine frozen backbones (ImageNet ResNet-18 and ViT-B, CLIP, MoCo-v3, MAE, DINO, DINOv2, R3M, Stable Diffusion), 50 demonstrations per task. The best backbone depended on the environment: ImageNet models were more stable on Meta-World while CLIP and DINOv2 underperformed there; DINO and DINOv2 did well on RoboCasa; R3M did well on SimplerEnv (Fig. 3). Predicting the true scene state from frozen features with a linear head correlated with control success (Pearson r = 0.753 on average), while ImageNet accuracy, segmentation and shape bias did not (Table 1).

### 3.18 Schäfer et al. (2023, workshop 2025). "Visual Encoders for Data-Efficient Imitation Learning in Modern Video Games". ALA Workshop 2025 (per the paper).

Opened: https://ar5iv.labs.arxiv.org/html/2312.02312

Measured (checked): BC in three games with 40 minutes to 8 hours of demonstrations. Frozen DINOv2 was the strongest in two games (Minecraft, Minecraft Dungeons) and worse than end-to-end encoders in CS:GO (Sec. 5.2, Table 3). For end-to-end encoders, 128x128 images were enough (Sec. 5.1). Pre-computing frozen DINOv2 ViT-L/14 embeddings cut training time from 280 to 4.6 minutes (Sec. 5.5, Table 5). With 10% of the data, pretrained and end-to-end encoders degraded similarly (Sec. 5.3).

### 3.19 Kumar, Raghunathan, Jones, Ma, Liang (2022). "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution". ICLR 2022 (oral, per arXiv comments).

Opened: https://arxiv.org/abs/2202.10054 (abstract)

Measured (checked, image classification, not robotics): over 10 distribution-shift datasets, full fine-tuning was about 2 points better than a linear probe in distribution but about 7 points worse out of distribution. Training the head first and then fine-tuning everything (LP-FT) beat plain fine-tuning by about 1 point in distribution and about 10 points out of distribution (abstract).

### 3.20 Model papers and model cards

- Caron et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers" (DINO). Listed as ICCV 2021, pp. 9650-9660 (the CVF page returned 403, so the venue is unchecked). Opened: https://arxiv.org/abs/2104.14294. Checked: self-supervised ViT features hold explicit segmentation information; k-NN 78.3% on ImageNet with a small ViT. One quote: "self-supervised ViT features contain explicit information about the semantic segmentation of an image" (abstract).
- Oquab et al. (2023, TMLR 2024). "DINOv2: Learning Robust Visual Features without Supervision". Opened: https://arxiv.org/abs/2304.07193, https://arxiv.org/html/2304.07193v2, https://mlanthology.org/tmlr/2024/oquab2024tmlr-dinov2 and https://github.com/facebookresearch/dinov2. Checked: trained on 142 million curated images; a 1.1-billion-parameter ViT-g distilled into smaller models (Sec. 3, 5); frozen features strong for segmentation and depth with linear heads, ahead of OpenCLIP (Sec. 7.4, Tables 10 to 11); ViT-S/14 has 384 dimensions, 6 heads, 12 blocks (Table 17); sizes 21M (ViT-S/14), 86M (ViT-B/14), 300M (ViT-L/14), 1,100M (ViT-g/14), Apache 2.0 licence (README).
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" (CLIP). ICML 2021, PMLR 139:8748-8763. Opened: https://proceedings.mlr.press/v139/radford21a.html. Checked: trained on 400 million image-text pairs.
- Siméoni et al. (2025). "DINOv3". arXiv:2508.10104, technical report. Opened: https://arxiv.org/abs/2508.10104. Checked (abstract): claims better dense features and a suite of model sizes. I found no control study of it apart from the real-robot test in 3.16.
- torchvision model cards (opened): ResNet-18 ImageNet weights: 11,689,512 parameters, 1.81 GFLOPS, 69.758% ImageNet top-1 (https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). ViT-B/16: 86,567,656 parameters, 17.56 GFLOPS (https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html). These "GFLOPS" figures count multiply-adds.

---

## 4. Synthesis by sub-question

### 4.1 Frozen or fine-tuned?

Evidence:
- Frozen features are cheap: features can be computed once, and BC training becomes at least ten times faster (Hansen 3.6; Schäfer 3.18: 280 to 4.6 minutes).
- Frozen features were clearly worse than fine-tuned or from-scratch ResNets on a 200-demonstration task (Chi 3.10: 0.58 against 0.92 and 0.94), and frozen encoders failed some real tasks completely (Theia 3.15; SpawnNet 3.14: 11.7%).
- Fine-tuning ResNets helped when combined with augmentation (Hansen 3.6: R3M 64.0 to 80.5) and a small learning rate (Chi 3.10).
- Fully fine-tuning big ViTs on small data hurt (VC-1 3.5: Adroit 59.3 to 15.9; Hansen 3.6: MVP 44.0 to 31.1; SpawnNet 3.14: 8.3%). Real-world gains from fine-tuning VC-1 were small (Silwal 3.8b: 59% against 57% for VC-1 Base, 47% against 45% for VC-1 Large).
- In image classification, fine-tuning can damage out-of-distribution accuracy, and training the head first (LP-FT) reduces this (Kumar 3.19).

My judgement: fine-tune a small CNN gently (lower backbone learning rate, head trained first, frozen BatchNorm). Keep large ViTs frozen.

### 4.2 ImageNet against robot-specific backbones (R3M, MVP, VC-1)

Evidence for robot-specific models: in frozen, few-demonstration simulation tests R3M led (R3M 3.2; Hu 3.7; VC-1 3.5), and MVP led frozen real-world tests against CLIP and supervised ImageNet (3.4).

Evidence against: under lighting, texture or distractor shifts they were not better than ImageNet models (Burns 3.9; Xie 3.13). With fine-tuning on real robots, ImageNet pretraining gave 41% against 28% for Ego4D, 33% for VC-1 and 25% for MVP (Dasari 3.8); in a ResNet-18 repeat R3M got 8% against 32% for ImageNet (Dasari App. D). On real Push-T, R3M (80%) lost to end-to-end training (95%) (Chi 3.10). On unseen real object instances, frozen R3M reached 15 to 20% (SpawnNet 3.14).

My judgement: robot-specific pretraining gives no reliable advantage for our case, and the ViT-L versions cost far more on a CPU.

### 4.3 Self-supervised ViTs (DINO, DINOv2) and CLIP

- DINO ViT-S/16 was the most robust frozen model under appearance shifts (Burns 3.9). DINOv2 ViT-S/14 did worse than expected in the same study, but well in other settings (Dong 3.17 on RoboCasa; Schäfer 3.18 in two of three games). Its frozen dense features are strong in vision benchmarks (Oquab 3.20).
- How features are read out matters: patch tokens beat the CLS token (Theia 3.15), and an attention-pooling head made frozen features much more robust (AFA 3.16).
- CLIP, frozen, was weaker than R3M and MVP for manipulation (R3M 3.2: 24% against 56% real; MVP 3.4: 18.8% on PickFruit) and MoCo (Parisi 3.1). Fine-tuned CLIP ViT-B/16 was the best encoder on Square (0.98, Chi 3.10), but that needs a GPU-scale budget for a ViT-B.

### 4.4 Out-of-distribution generalisation and emergent segmentation

Evidence: for ViTs, emergent segmentation predicted robustness better than ImageNet accuracy (Burns 3.9). A 2026 preprint found that predicting the true scene state from frozen features predicted control success better than segmentation, ImageNet accuracy or shape bias (Dong 3.17). Augmentation during policy training was a strong lever against appearance shifts (Hansen 3.6, Table 3; Xie 3.13; Mandlekar 3.11).

My judgement: all of this concerns appearance (lighting, texture, distractors, camera pose). It does not show which backbone handles new track shapes.

### 4.5 How much does pretraining help with a few hundred demonstrations?

Evidence:
- 10 demonstrations: frozen pretrained features slightly better than LfS (Hansen 3.6, Fig. 8). 100: LfS generally better (same figure).
- 136 demonstrations (real Push-T): end-to-end ResNet-18 trained from scratch 95% against R3M 80% and ImageNet 15% (Chi 3.10).
- 200 demonstrations (Square): ResNet-18 from scratch 0.94, fine-tuned ImageNet 0.92, frozen ImageNet 0.58 (Chi 3.10).
- 50 demonstrations with a ViT: no pretraining gave 0% on all real tasks, so ViTs need pretraining (Dasari 3.8). A ViT-B trained from scratch also failed on Square (0.22, Chi 3.10).
- Fewer demonstrations needed: MVP matched CLIP with half the demonstrations (Radosavovic 3.4).

My judgement: at our scale (208 demonstrations, about 18,500 pictures, close to Square's 200 demonstrations), pretraining is not what decides success for a small CNN; augmentation and data variety are. Pretraining is still a cheap insurance for appearance changes and faster training.

### 4.6 What the literature does not cover

- New track geometry with the same look (our main exam shift). No study above tests it.
- A camera on the forearm that rotates with the forearm roll. Most studies use a fixed third-person camera, some a wrist camera (MVP 3.4).
- Stop-and-go control with discrete move classes. Most studies regress continuous actions.
- CPU-only training. Every study used GPUs.

---

## 5. Cost on a laptop CPU (calculations and guesses, not measurements)

Cost per 224x224 picture, in multiply-adds (GMACs; torchvision calls these GFLOPS):

| Backbone | Parameters | GMACs per picture | Source |
|---|---|---|---|
| ResNet-18 | 11.7M | 1.81 | torchvision card (checked) |
| DINO ViT-S/16 | about 22M (Radosavovic 3.4 gives 22M for ViT-S) | about 4.6 (calculation) | 197 tokens, 384 dims, 12 blocks |
| DINOv2 ViT-S/14 | 21M | about 6.1 (calculation) | 257 tokens, 384 dims, 12 blocks |
| ViT-B/16 (CLIP, MVP-B) | 86.6M | 17.56 | torchvision card (checked) |
| ViT-L/16 (VC-1, MVP-L) | 307M | about 61 (calculation); MVP paper says about 64 GFLOPs | Radosavovic 3.4 |

My calculation counts, per block, the query-key-value projection, the attention matrix, the output projection and a 4x MLP. For ViT-B/16 it gives 17.6, which matches torchvision's 17.56, so the method seems sound.

Guess (not measured): if PyTorch on the 6-core laptop sustains 50 to 200 GMAC/s, then for 18,500 pictures:

| Job | ResNet-18 | ViT-S/14 | ViT-L/16 |
|---|---|---|---|
| Compute frozen features once | about 3 to 11 min | about 9 to 38 min | about 1.6 to 6 h |
| Fine-tune 20 epochs at 224x224 (forward plus backward, about 3 times forward) | about 3 to 11 h | about 9 to 38 h | days (impractical) |
| Fine-tune 20 epochs at 160x120 | about 1 to 4 h | not computed | not computed |
| One picture at run time | about 10 to 40 ms | about 30 to 120 ms | about 0.3 to 1.2 s |

At 1.2 s per step now, a ResNet-18 or ViT-S adds little to finishing time; a ViT-L could add a quarter or more. Before choosing, time 100 forward and backward passes on the laptop; these numbers are only guesses.

Note: if features are computed once and frozen, image augmentation is lost unless several augmented copies are also computed. This is one reason fine-tuning a small CNN is attractive.

---

## 6. Recommendation (my judgement, not evidence)

1. **Default backbone: ImageNet-pretrained ResNet-18 (torchvision weights), fine-tuned end-to-end.**
   - Learning rate for the backbone about 10 times lower than for the head (as in Chi 3.10); optionally train the head first with the backbone frozen, then unfreeze (LP-FT, Kumar 3.19).
   - Frozen BatchNorm statistics (as in ACT 3.12), because CPU batches will be small; or GroupNorm if training from scratch (Chi 3.10).
   - Keep spatial information: use the layer-3 or layer-4 feature map with spatial softmax or a small convolution and attention head, not only the global average (Parisi 3.1, Theia 3.15, robomimic 3.11).
   - Augment: small random shifts or crops (Hansen 3.6, robomimic 3.11, Xie 3.13) and colour jitter (Hansen 3.6, Table 3). Avoid flips and rotations unless the move label is transformed with the picture.
   - Dropout in the head (Dasari 3.8).
   - Lower input resolution (for example 224x168 or 160x120) if the 1 cm line still spans several pixels; check this on our pictures first.
2. **One comparison: frozen DINO ViT-S/16 (or DINOv2 ViT-S/14) patch tokens with a small attention-pooling head**, features computed once (cheap). DINO was the most robust model in Burns et al.; DINOv2 has newer evidence but a weak result in the same study.
3. **Not recommended on our laptop:** VC-1 and MVP ViT-L (61 to 64 GMACs per picture, no robustness advantage in Burns et al., collapse under full fine-tuning in Majumdar et al.); R3M as the main choice (weak under shift and in real fine-tuned tests); CLIP frozen (weak for manipulation), CLIP fine-tuned (ViT-B cost).
4. **How to decide:** leave-one-sheet-out validation (train on five sheets, test on the sixth, rotate). This is the nearest stand-in for unseen tracks; accuracy on the training sheets will not tell the backbones apart (Dasari 3.8 and Burns 3.9 both found in-distribution success a poor guide). A quick linear probe that predicts puck position or line direction from frozen features can rank candidates before full training (Hu 3.7, Dong 3.17).
5. Expect the backbone choice to matter less than the variety of tracks, the augmentation and the action design (Hansen 3.6, Xie 3.13). This expectation is a guess.

---

## 7. Claims list

| id | claim | source | location | status | kind |
|---|---|---|---|---|---|
| pretrained_representations-1 | Frozen pretrained models beat random features and a from-scratch ConvNet in BC across Habitat and MuJoCo | Parisi et al. 2022 | Sec. 4.1, Fig. 6 | checked | measured |
| pretrained_representations-2 | No single model best in all domains; MoCo beat supervised ImageNet and CLIP on average | Parisi et al. 2022 | Sec. 4.1, Fig. 6 | checked | measured |
| pretrained_representations-3 | Early layers better for fine control, last layer for navigation; MoCo layers 3 to 5 matched or beat true state | Parisi et al. 2022 | Sec. 4.4 to 4.5, Figs. 9 to 10 | checked | measured |
| pretrained_representations-4 | Crop-only pretraining augmentation as good as or better than others; authors suggest it emphasises relative position | Parisi et al. 2022 | Sec. 4.3, Fig. 8 | checked | speculation |
| pretrained_representations-5 | Parisi et al. do not mention augmentation for their from-scratch baseline, so the scratch gap may partly reflect missing augmentation (absence and interpretation are mine) | Parisi et al. 2022 | Sec. 3.2, App. | unchecked | speculation |
| pretrained_representations-6 | Frozen R3M averaged 62.4% on 12 sim tasks, over 20 points above scratch and over 10 above CLIP and MoCo | Nair et al. 2022 | Sec. 4.3, Table 1 | checked | measured |
| pretrained_representations-7 | Real Franka, 20 demos per task: R3M 56% against CLIP 24% | Nair et al. 2022 | Table 3 | checked | measured |
| pretrained_representations-8 | Frozen MVP up to 75% better than CLIP and 81% better than supervised ImageNet and scratch on a real robot, 80 demos | Radosavovic et al. 2022 | Abstract, Sec. 1, Fig. 4 | checked | measured |
| pretrained_representations-9 | ViT-L costs about 64 GFLOPs per 224 px image, about 15 times ResNet-50 | Radosavovic et al. 2022 | Sec. 3.1 | checked | definition |
| pretrained_representations-10 | Fine-tuning MVP ViT-S and CLIP ResNet-50 gave only marginal real-robot gains | Radosavovic et al. 2022 | Sec. 5.5 | checked | measured |
| pretrained_representations-11 | No PVR best on all CortexBench benchmarks; VC-1 best mean (68.7) against MVP-L 67.5, R3M 58.0, CLIP 57.0 | Majumdar et al. 2023 | Sec. 4 to 5, Tables 2, 4 | checked | measured |
| pretrained_representations-12 | Full fine-tuning of VC-1 collapsed in few-demo imitation (Adroit 59.3 to 15.9, Meta-World 88.8 to 22.7, DMC 66.9 to 6.7); MAE adaptation helped | Majumdar et al. 2023 | Sec. 6, Table 5 | checked | measured |
| pretrained_representations-13 | Authors attribute the collapse to overfitting 307M parameters on at most about 50K frames | Majumdar et al. 2023 | Sec. 6 | checked | speculation |
| pretrained_representations-14 | Real Franka: VC-1 frozen 70%, MAE-adapted 85%, MVP 55% | Majumdar et al. 2023 | Sec. 7 | checked | measured |
| pretrained_representations-15 | A shallow ConvNet trained from scratch with random-shift augmentation was competitive with frozen PVR, MVP and R3M | Hansen et al. 2023 | Table 2, Fig. 4 | checked | measured |
| pretrained_representations-16 | With 10 demos frozen models were marginally better; with 100 demos learning from scratch was generally better | Hansen et al. 2023 | Fig. 8 | checked | measured |
| pretrained_representations-17 | Fine-tuning helped ResNets only with augmentation (R3M 64.0 to 80.5); fine-tuning the ViT (MVP) hurt (44.0 to 31.1) | Hansen et al. 2023 | Table 5 | checked | measured |
| pretrained_representations-18 | Under colour and background shifts, LfS with augmentation (53.1, 46.6) beat the best frozen model (37.2, 9.2) | Hansen et al. 2023 | Table 3 | checked | measured |
| pretrained_representations-19 | Frozen features make BC training at least 10 times faster; LfS is faster at inference | Hansen et al. 2023 | Compute section, Table 4 | checked | measured |
| pretrained_representations-20 | Under frozen BC on 21 tasks, R3M was best overall, ImageNet-only VICRegL close; ResNets generally beat ViTs | Hu et al. 2023 | Sec. 4.2, Fig. 7 | checked | measured |
| pretrained_representations-21 | The value of a pretrained model depends heavily on the policy learning method | Hu et al. 2023 | Abstract, Sec. 4.1 | checked | measured |
| pretrained_representations-22 | A linear probe of the true state predicts BC success (Pearson r = -0.78 for probe loss) | Hu et al. 2023 | Sec. 4.4, Fig. 9 | checked | measured |
| pretrained_representations-23 | Fine-tuned MAE ViT, 50 real demos: ImageNet 41%, Ego4D 28%, VC-1 33%, MVP 25% (standard errors about 7 points) | Dasari et al. 2023 | Table 1 | checked | measured |
| pretrained_representations-24 | No pretraining (ViT) gave 0% on all three real tasks | Dasari et al. 2023 | Table 1 | checked | measured |
| pretrained_representations-25 | Doubling pretraining data did not help; image distribution mattered more | Dasari et al. 2023 | Table 2 | checked | measured |
| pretrained_representations-26 | Simulation results barely predicted real results (R squared 0.32) | Dasari et al. 2023 | Fig. 5 | checked | measured |
| pretrained_representations-27 | Dropout p = 0.2 in the policy almost consistently improved real-world success | Dasari et al. 2023 | Sec. 5, Fig. 4 | checked | measured |
| pretrained_representations-28 | ResNet-18 SimCLR repeat: R3M 8% against ImageNet 32% real | Dasari et al. 2023 | App. D | checked | measured |
| pretrained_representations-29 | With aligned sim and real set-ups, sim rankings predicted real rankings (SRCC 0.895; intro says 0.929) | Silwal et al. 2023 | Sec. IV-B | checked | measured |
| pretrained_representations-30 | Fine-tuning plus augmentation gave small real gains for VC-1 (59% against 57%; 47% against 45%) | Silwal et al. 2023 | Sec. IV-D, Table IV | checked | measured |
| pretrained_representations-31 | Manipulation-specific models did not generalise better under visual shift; the best ranked seventh of 15 | Burns et al. 2024 | Fig. 4 | checked | measured |
| pretrained_representations-32 | ViT-DINO was the most robust frozen model; six of the top seven were ViTs | Burns et al. 2024 | Sec. 4 | checked | measured |
| pretrained_representations-33 | For ViTs, emergent segmentation (Jaccard of CLS attention) strongly predicted robustness; ImageNet accuracy did not | Burns et al. 2024 | Sec. 5.1, Fig. 5, App. A.8 | checked | measured |
| pretrained_representations-34 | DINOv2 ViT-S/14 did worse than expected in and out of distribution | Burns et al. 2024 | App. A.1 | checked | measured |
| pretrained_representations-35 | Real ALOHA under shift (50 demos): MoCo-v3 40%, MVP 0% | Burns et al. 2024 | Table 1, Sec. 5.4 | checked | measured |
| pretrained_representations-36 | Authors argue augmentation choice matters more than supervision | Burns et al. 2024 | Sec. 4 | checked | speculation |
| pretrained_representations-37 | Square (200 demos): ResNet-18 scratch 0.94, frozen 0.58, fine-tuned 0.92; CLIP ViT-B/16 0.22, 0.70, 0.98 | Chi et al. 2023 | Sec. 5.4, Table 5 | checked | measured |
| pretrained_representations-38 | Fine-tuning with a 10 times smaller learning rate gave the best results | Chi et al. 2023 | Sec. 5.4 | checked | measured |
| pretrained_representations-39 | Real Push-T (136 demos): end-to-end 95%, R3M 80%, ImageNet 15% | Chi et al. 2023 | Sec. 6.1, Table 6 | checked | measured |
| pretrained_representations-40 | The released real-pretrained config freezes an ImageNet ResNet-18, so the Push-T pretrained runs were probably frozen | Chi et al. 2023 (code) | config file | unchecked | speculation |
| pretrained_representations-41 | ResNet-18 plus spatial softmax on 200 demos; removing pixel-shift augmentation cut Square success by 47% | Mandlekar et al. 2021 | Sec. 3.2, 4.3, Fig. 2a, App. D.1 | checked | measured |
| pretrained_representations-42 | ACT trains torchvision-pretrained ResNet-18 backbones with frozen BatchNorm end-to-end on 50 demos | Zhao et al. 2023 | Sec. IV-C, V-B; backbone.py | checked | definition |
| pretrained_representations-43 | Frozen R3M and CLIP did not close the generalisation gap compared with a scratch CNN | Xie et al. 2023 | Sec. 5.2, Figs. 7, 11 | checked | measured |
| pretrained_representations-44 | New camera pose was the hardest real shift (45.8% against 91.7%); random crops helped most | Xie et al. 2023 | Sec. 5.1 to 5.2, Figs. 1, 6 | checked | measured |
| pretrained_representations-45 | Unseen real instances: SpawnNet 86.7%, LfS with augmentation 66.7%, R3M 20%; frozen DINO 11.7%, full fine-tune 8.3% | Lin et al. 2023 | Tables 3, 6 | checked | measured |
| pretrained_representations-46 | Patch tokens consistently beat the CLS token for robot learning | Shang et al. 2024 | Sec. 4.2, Table 3c | checked | measured |
| pretrained_representations-47 | Frozen encoders got 0% on a real task where fine-tuning worked | Shang et al. 2024 | App. D.3.1 | checked | measured |
| pretrained_representations-48 | Attention pooling over frozen patch features improved robustness; real pushing under shift 0% to 100% (16 rollouts) | Tsagkas et al. 2025 | Sec. V-B, Table II | checked | measured |
| pretrained_representations-49 | Best frozen backbone depended on the environment; a state-prediction probe predicted control (r = 0.753) | Dong et al. 2026 | Fig. 3, Table 1 | checked | measured |
| pretrained_representations-50 | Frozen DINOv2 best in two of three games; 128x128 enough; cached features cut training from 280 to 4.6 min | Schäfer et al. 2023 | Sec. 5.1 to 5.5, Tables 3, 5 | checked | measured |
| pretrained_representations-51 | Fine-tuning was about 7 points worse out of distribution than a linear probe; LP-FT about 10 points better than fine-tuning | Kumar et al. 2022 | Abstract | checked | measured |
| pretrained_representations-52 | Self-supervised ViT (DINO) features contain explicit segmentation information | Caron et al. 2021 | Abstract | checked | measured |
| pretrained_representations-53 | DINOv2 frozen features strong for segmentation and depth; ViT-S/14 21M parameters, Apache 2.0 | Oquab et al. 2024; DINOv2 README | Sec. 7.4, Tables 10 to 11, 17; README | checked | definition |
| pretrained_representations-54 | ResNet-18: 11.7M parameters, 1.81 GFLOPS; ViT-B/16: 86.6M, 17.56 GFLOPS | torchvision model cards | weights tables | checked | definition |
| pretrained_representations-55 | DINOv2 ViT-S/14 costs about 6.1 GMACs and ViT-L/16 about 61 GMACs per 224 px image | my calculation | section 5 | unchecked | speculation |
| pretrained_representations-56 | On the laptop, fine-tuning ResNet-18 for 20 epochs would take about 3 to 11 h at 224 px (assumed 50 to 200 GMAC/s) | my calculation | section 5 | unchecked | speculation |
| pretrained_representations-57 | CLIP learns image features by matching 400 million internet image-text pairs | Radford et al. 2021 | Abstract | checked | definition |

---

## 8. Conclusion for the project (also returned to the orchestrator)

Evidence: no pretrained backbone wins everywhere (Majumdar 2023; Hu 2023). Robot-specific models (R3M, MVP, VC-1) shine mainly in frozen, few-demonstration simulation tests; they were not more robust to lighting, texture or distractor shifts (Burns 2024) and did not beat ImageNet in fine-tuned real-robot tests (Dasari 2023). With 100 to 200 demonstrations, a ResNet-18 trained end-to-end with shift augmentation matched or beat frozen features (Hansen 2023; Chi 2023: Square 0.94 scratch, 0.92 fine-tuned, 0.58 frozen; real Push-T 95% end-to-end, 80% R3M, 15% ImageNet). Fully fine-tuning big ViTs on small data hurt (VC-1 Adroit 59 to 16). Frozen DINO ViT-S was the most robust to appearance shifts; DINOv2 ViT-S underperformed in the same study. No study tested unseen track shapes.

Judgement: the safest choice is an ImageNet ResNet-18 (11.7M parameters, 1.8 GMACs per 224 px picture), fine-tuned on the CPU with a lower backbone learning rate, frozen BatchNorm, shift and colour augmentation, and spatial feature maps. Compare once with frozen DINO ViT-S/16 patch tokens (cheap if cached). Choose by leave-one-sheet-out tests, not training-sheet accuracy. Skip ViT-L models (VC-1, MVP-L).

## 9. Seen only as citations (not opened)

- Geirhos et al. 2021 (shape bias and robustness), cited by Burns et al.
- KitchenShift (Xing et al. 2021), cited by Burns et al. for the Franka Kitchen shifts.
- DrQ and DrQ-v2 (Kostrikov et al.; Yarats et al.), cited by Hansen et al. for random-shift augmentation.
- VIP (Ma et al. 2023), evaluated by Burns et al., Hu et al. and Majumdar et al.; not opened.
- He et al. 2016 (ResNet) and He et al. 2022 (MAE): I used the torchvision model cards instead.
- PIE-G (Yuan et al. 2022) and Träuble et al. 2022 on pretrained representations for generalisation in RL: not opened, not used.
