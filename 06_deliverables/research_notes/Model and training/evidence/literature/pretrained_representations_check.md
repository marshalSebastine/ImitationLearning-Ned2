# Sceptical check: pretrained vision backbones for control

Key: `pretrained_representations`. Checker's note, 9 November 2026.

## How this check was done

- I opened every cited source myself with WebFetch: the ar5iv or arXiv HTML pages, the arXiv abstract pages (for venues), the GitHub raw files, the torchvision model cards, and the PMLR pages for CLIP and Burns et al. WebFetch passes the page to a small reading model, so for every number I asked for the actual table rows and compared them with the claim digit by digit.
- One page (Parisi et al. on ar5iv) was also saved as `sources/parisi/page.html` in this folder during a connection test. I searched its full text by keyword for claim 5. Nothing else was saved.
- Bar charts are images and I could not read them. Where a claim depends on bar heights, I say so.
- Verdicts: supported (the source says this), corrected (the source says something different; the correct statement is given), not supported, could not open.
- Tally: 51 supported (several with caveats), 6 corrected (claims 3, 29, 32, 43, 44, 45), 0 not supported, 0 could not open.
- Quotes are avoided. Everything below is paraphrase unless marked.

## Verdicts per claim

### Parisi, Rajeswaran, Purushwalkam, Gupta (2022), ICML 2022, PMLR 162:17359-17371
URL opened: https://ar5iv.labs.arxiv.org/html/2203.03580 (venue and pages confirmed by the arXiv Journal-ref at https://arxiv.org/abs/2203.03580)

- **1, supported, with caveats.** Sec. 4.1 and the Fig. 6 caption say that any pretrained representation is clearly better than frozen random features and than training the perception module from scratch. The text adds that this holds in the small-dataset regime studied. The text never says that this holds in each of the four domains separately. That rests on the Fig. 6 bars, which I could not read. Demonstrations: MuJoCo tasks used 25 to 100 per task (Sec. 3.3; App. A.2: DMC 100, Adroit 100, Kitchen 25). Habitat used 10,000 trajectories per scene, about 2.1M samples, so "25 to 100" applies to MuJoCo only, as the claim says. Encoders stay frozen during policy learning (Sec. 5, Fig. 1 caption). All four domains are simulated, and all use behaviour cloning. Models: supervised ResNet-34 and ResNet-50 (ImageNet), MoCo ResNet-50, and CLIP ResNet-50 and ViT.
- **2, supported.** Sec. 4.1: no representation is clearly best across all four domains. On average the self-supervised MoCo beats the "SL" models; the paper puts CLIP in that group with RN50. Fig. 6 caption: MoCo matches ground-truth state in Habitat, but no off-the-shelf model matches it in MuJoCo.
- **3, corrected.** Sec. 4.4: early convolution layers work better for fine MuJoCo control, even matching or beating ground truth there. Fig. 9: later layers work better in Habitat. Sec. 4.5: layer-3 features are competitive with ground truth in MuJoCo, and layer 5 is strong in Habitat. Two corrections are needed. (a) The combined layers 3+4+5 model is the authors' own MoCo trained with crop-only augmentation, not off-the-shelf MoCo. (b) The Fig. 10 caption calls it competitive with ground-truth state in all domains and better only in Kitchen. Sec. 4.5 adds "sometimes even better". Correct statement: competitive with ground truth in all four domains, and better than it in Kitchen.
- **4, supported (authors' speculation).** Sec. 4.3, Fig. 8: in MoCo pretraining, crop-only augmentation was as good as or better than all other choices. Colour-only did worse everywhere except DMC. The authors hypothesise that crop invariance highlights the relative displacement between agent and objects rather than absolute image position. Note that this concerns augmentation during pretraining, not during policy training.
- **5, supported (I checked the factual part; the interpretation is still speculation).** I searched the full saved text for augment, crop, jitter and shift. Every hit concerns MoCo pretraining, related work, or the conclusion. The From Scratch baseline description (Sec. 3.2) and App. A.1 and A.2 say nothing about augmentation. Supporting evidence for the interpretation: Hansen et al. (2023, Table 2) show from-scratch behaviour cloning jumping from 16.9 to 51.4 on Adroit Relocate once random shift is added.

### Nair, Rajeswaran, Kumar, Finn, Gupta (2022), R3M, CoRL 2022
URL opened: https://ar5iv.labs.arxiv.org/html/2203.12601 (CoRL 2022 confirmed by the arXiv comments; PMLR pages not checked)

- **6, supported, with a caveat.** Sec. 4.2: 5, 10 and 25 demonstrations in Meta-World and Franka Kitchen, and 25, 50 and 100 in Adroit. There are 12 tasks (5 Meta-World, 5 Kitchen, 2 Adroit), with results averaged over 3 views, 3 dataset sizes and 3 seeds. Table 1 (R3M column, all domains): 62.4 ± 1.3. R3M is a ResNet-50 trained on Ego4D (Sec. 3.2 to 3.3). It is frozen: Sec. 4.1 studies frozen perception, and App. B.4 says only the Scratch baseline passes gradients into the encoder. The caveat: the paper says R3M beats scratch by more than 20% and all prior representations (CLIP, MoCo) by more than 10%. It does not say whether these are percentage points. The per-method bars (Fig. 4) are an image, so "points" is unverified.
- **7, supported.** Table 3, 20 demonstrations per task, 10 trials each. R3M scored 80, 30, 60, 70 and 40, averaging 56%. CLIP scored 70, 10, 0, 40 and 0, averaging 24%.

### Radosavovic, Xiao, James, Abbeel, Malik, Darrell (2022), CoRL 2022
URL opened: https://ar5iv.labs.arxiv.org/html/2210.03109 (CoRL 2022 confirmed by the arXiv comments)

- **8, supported.** Abstract: up to 75% better than CLIP, and up to 81% better than both supervised ImageNet and training from scratch. These are task maxima, not averages. Sec. 4: xArm7, a first-person wrist-mounted RealSense camera, 80 demonstrations per task. Sec. 3.2: encoders frozen. Sec. 5.1 and Fig. 5: MVP reaches CLIP's performance with 50% fewer demonstrations. The paper says "comparable", which is close to "matched".
- **9, supported.** Sec. 3.1: ViT-L has 307M parameters and needs about 64 gigaflops at 224x224, about 15 times a ResNet-50. Cross-check with the torchvision ResNet-50 card (25.6M parameters, 4.09 GFLOPS): 64/4.09 is about 15.6 for compute, but 307/25.6 is only about 12 for parameters. So Sec. 4's "about 15 times as many parameters" is loose. My own count for ViT-L/16 is 61.6 G multiply-adds.
- **10, supported.** Sec. 3.2: encoders are frozen to stop large models overfitting and to cut GPU memory and training time. Sec. 5.5: fine-tuning ViT-S HoI and CLIP ResNet-50 improves results only marginally and does not close the gap. The numbers are only in a figure.

### Majumdar et al. (2023), VC-1 and CortexBench, NeurIPS 2023 (venue not checked)
URL opened: https://ar5iv.labs.arxiv.org/html/2303.18240

- **11, supported.** Table 2, frozen encoders. R3M (ResNet-50) is best on Adroit 73.3, Meta-World 96.0 and DMC 81.1. Sec. 4: no single model is best on every benchmark. Mean success: MVP ViT-L 67.5, R3M 58.0, CLIP (ViT-B) 57.0. Table 4: VC-1 (ViT-L) 68.7. Note: VC-1's lead comes from navigation, TriFinger and mobile pick. On the three few-demonstration behaviour-cloning benchmarks, VC-1 (59.3, 88.8, 66.9) is far below R3M.
- **12, supported.** Table 5, frozen then end-to-end then MAE adaptation. Adroit: 59.3 ± 5.2, 15.9 ± 9.8, 72.0 ± 3.3. Meta-World: 88.8 ± 2.2, 22.7 ± 9.7, 96.0 ± 1.8. DMC: 66.9 ± 1.4, 6.7 ± 3.8, 80.9 ± 1.8.
- **13, supported (authors' hypothesis).** Sec. 6: they hypothesise overfitting from fine-tuning a 307M-parameter model on a small dataset of at most 50K frames.
- **14, supported.** The numbers are in Table 5 (Franka column; 4 tasks with 10 trials each) and are discussed in Sec. 7. MVP ViT-L 55.0 ± 10.4, VC-1 frozen 70.0 ± 10.9, VC-1 with MAE adaptation 85.0 ± 9.1. The standard errors are about 10 points.

### Hansen, Yuan, Ze, Mu, Rajeswaran, Su, Xu, Wang (2023), ICML 2023 (venue not on the arXiv page; not checked)
URLs opened: https://ar5iv.labs.arxiv.org/html/2212.05749 and https://arxiv.org/html/2212.05749v2

- **15, supported.** The from-scratch network is a shallow ConvNet with 4 to 6 layers (Sec. 2; App. B.1 lists five convolutions with BatchNorm). It uses random shift with padding 12, on 256x256 images centre-cropped to 224. The pretrained models are all frozen: PVR (MoCo-v2 ResNet-50, ImageNet), MVP (ViT-S, HoI) and R3M (ResNet-50, Ego4D). Table 2: Adroit Relocate, from scratch with augmentation 51.4 ± 7.7 against 47.5 ± 2.6 for the best pretrained model (16.9 without augmentation). Real Pick: 55.0 ± 5.0 against 35.0 ± 15.0, from 20 trials over 2 seeds with 20 demonstrations (reach used 10). Location note: Fig. 4 shows PixMC with PPO (simulated RL), not the real robot. Pretrained models still won 4 of the 9 rows (Pen, Finger Spin, Cheetah Run, Real Reach).
- **16, supported.** Sec. 4 and Fig. 8: 10, 25 and 100 demonstrations, averaged over the DMControl tasks and 3 seeds. Frozen models do marginally better at 10, and from scratch is generally better at 100. There are no numbers in the text.
- **17, supported.** Table 5 (Adroit, 5 seeds). PVR: 52.9 frozen, 50.5 fine-tuned without augmentation, 65.0 fine-tuned with augmentation. MVP: 44.0, 18.7, 31.1. R3M: 64.0, 55.5, 80.5. From scratch: 19.4, or 63.1 with augmentation. Fine-tuned R3M with augmentation beat from-scratch-with-augmentation by about 17 points. Fine-tuned ImageNet MoCo with augmentation only tied it (65.0 against 63.1).
- **18, supported (pairing clarified).** Table 3 shows behaviour-cloning policies on 4 DMC tasks with 5 seeds (the setting is given in the Fig. 6 caption). Random colours: jitter 53.1, overlay 39.3, best pretrained 37.2. Video backgrounds: jitter 33.0, overlay 46.6, best pretrained 9.2. So 53.1 belongs to random colours and 46.6 to video backgrounds. Both from-scratch variants beat the best frozen model on both test sets.
- **19, supported.** The text says behaviour-cloning updates are at least an order of magnitude faster with cached frozen features, and that inference favours the from-scratch models. The Table 4 headers were garbled in extraction. The values as read were 0.263 vs 0.003 and 0.270 vs 0.006 s per iteration for training, and 1.61 vs 2.66 and 3.81 vs 11.00 s per episode for inference. I am not sure the columns are mapped right.

### Hu, Wang, Li, Gao (2023), ICML 2023 (venue not on the arXiv page; not checked)
URL opened: https://ar5iv.labs.arxiv.org/html/2304.04591

- **20, supported.** Sec. 3.3: 21 tasks (Meta-World 8, Robosuite 8, Franka Kitchen 5). App. C: 1 to 25 demonstrations in Meta-World and Kitchen, 5 to 50 in Robosuite. Sec. 3.1: frozen encoders. Sec. 4.2: R3M best overall, the ImageNet-only VICRegL close behind, ResNets tending to beat ViTs, and no model best in all three environments.
- **21, supported.** Abstract: the benefit of pretraining depends heavily on the downstream method (RL, behaviour cloning, imitation with a visual reward function). RL-based evaluation is highly variable and unreliable.
- **22, supported.** Sec. 4.4 and Fig. 9: Pearson -0.78 and Spearman -0.73, for Franka Kitchen only (5 tasks). No values are given for the other environments.

### Dasari, Srirama, Jain, Gupta (2023), CoRL 2023 (venue not checked)
URL opened: https://arxiv.org/html/2310.09289

- **23, supported, but the reader needs one clarification.** Table 1 real averages: ImageNet 41 ± 7.1, Ego4D 28 ± 6.9, Kinetics 44 ± 7.1, VC-1 33 ± 6.9, MVP 25 ± 6.6. Set-up: 50 tele-operated demonstrations per task, 3 real tasks (block stacking, pouring, toasting), a third-person camera, and the whole network fine-tuned end-to-end (Sec. 3). The clarification: "ImageNet" here means self-supervised MAE on 1M ImageNet images, without labels, with a ViT-B-sized encoder (12 layers, 12 heads, 768 dims; App. A, Table 3). It is not supervised ImageNet weights. Kinetics (44) was slightly above ImageNet (41).
- **24, supported.** Table 1: no pretraining gives 0% on all three real tasks. Simulation average is 38% (RoboSuite 2, Meta-World 72, Kitchen 40).
- **25, supported (wording).** Table 2: Soup-1M (200K images from each of 5 datasets) scored 41 ± 7.1, and Soup-2M scored 41 ± 7.0. Soup-1M plus 1M frames from 100 Days of Hands scored 60 ± 6.7, at the same 2M total size.
- **26, supported.** Fig. 5: R squared is 32% across models (34% for the best-matched sim and real task pair).
- **27, supported.** App. B: dropout p = 0.2 after the ReLUs of the two MLP layers in the policy head. Sec. 5 says it almost consistently helps on the real robot. The Fig. 4 caption says it often hurts in simulation. Block stacking with a human pushing the cube out of the gripper rose from 10% to 24%.
- **28, supported.** App. D (ResNet-18, SimCLR), real averages: off-the-shelf R3M 8 ± 3.8, ImageNet 32 ± 6.9, 100 Days of Hands 44 ± 7.1. Also Ego4D 25, Kinetics 31, RoboNet 21. I saw no from-scratch ResNet-18 row.

### Silwal et al. (2023, rev. 2024), arXiv 2310.02219v2 (no venue on the page)
URL opened: https://arxiv.org/html/2310.02219v2

- **29, corrected.** The paper reports the "Sim2Real Correlation Coefficient" (SRCC), borrowed from Kadian et al. It never says Spearman and gives no formula. The coefficient is computed over (model, task) points (Fig. 3 caption), not over model rankings. The models are R3M, CLIP, MVP, VC-1 Base and VC-1 Large. The values themselves are right: 0.929 in the Introduction, and in Sec. IV-B -0.475 for CortexBench against real and 0.895 for aligned simulation against real.
- **30, supported, with a caveat.** Table IV, real "all tasks" averages: VC-1 Base 57 frozen without augmentation, 59 fine-tuned with augmentation. VC-1 Large: 45 and 47. Demonstrations (Sec. III): 30 for TriFinger, Reach and Pick, and 150 for Open Drawer. The caveat: the averages do not equal the mean of the four task columns (row 1 simulation values 40, 97, 80, 57 average 68.5, but the table lists 55), and the paper does not explain this. The other rows show frozen with augmentation (46, 40) and fine-tuned without augmentation (VC-1 Base 37) both doing worse.

### Burns, Witzel, Hamid, Yu, Finn, Hausman (2023; CoRL 2024, PMLR 270:4525-4545, published 2025)
URLs opened: https://arxiv.org/html/2312.12444v1, https://arxiv.org/html/2312.12444 (also served v1), https://arxiv.org/abs/2312.12444, https://proceedings.mlr.press/v270/burns25a.html (venue and pages confirmed)

- **31, supported.** 15 models. A 2-layer MLP on frozen features trained with 10 demonstrations; the text does not say "per task". FrankaKitchen and Meta-World, 10 tasks. Lighting, texture and distractor shifts. Fig. 4 caption: the best manipulation-specific model ranks seventh.
- **32, corrected.** Sec. 4 says the DINO-trained model (listed as ViT-S/16 in Table 2) beats all three manipulation models under every shift except Meta-World distractors, and that six of the seven best out-of-distribution models are ViTs. Both parts are supported. The text never says ViT-DINO was the most robust model overall. Any first-place ranking would be in the Fig. 4 image, which I could not read.
- **33, supported.** Sec. 5 and App. A.8: Jaccard index of the [CLS]-token attention against PASCAL VOC val masks. It has a strong positive link with out-of-distribution success among ViTs. ImageNet linear-probe accuracy predicts poorly. Shape bias looks promising only for ResNets. No numeric correlations appear in the text. The "last layer" detail was not confirmed in what I read.
- **34, supported.** App. A.1 and Table 2: DINOv2 ViT-S/14 (21M parameters, LVD-142M) did worse in and out of domain. It is called surprising, with no explanation.
- **35, supported for v1, with a caveat.** Sec. 5.4 and Table 1 (v1): ALOHA, 50 expert episodes, ViT-B encoders inside ACT, picking up a screwdriver, with object placement and lighting direction shifted, 10 rollouts. MoCo-v3 reached 40% and MVP 0%. The text does not say whether the encoders were frozen. The published CoRL abstract says segmentation score predicts real ALOHA performance, so the final version may contain more real-robot models. I did not check that version's table.
- **36, supported (authors' conclusion).** Sec. 4: the choice of augmentations outweighs supervision. This is based on MoCo and DINO against MAE, and on a ViT trained on Stylized ImageNet.

### Chi, Feng, Du, Xu, Cousineau, Burchfiel, Song (2023), Diffusion Policy, RSS 2023 (extended arXiv v5, 14 March 2024)
URLs opened: https://arxiv.org/html/2303.04137 (both halves); config https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/config/train_diffusion_unet_real_pretrained_workspace.yaml; workspace code https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py

- **37, supported.** Table 5 (Square, proficient human, 200 demonstrations per Table 3; CNN-based diffusion policy), from scratch / frozen / fine-tuned. ResNet-18 (ImageNet-21k): 0.94 / 0.58 / 0.92. ResNet-34: 0.92 / 0.40 / 0.94. ViT-B/16 (CLIP): 0.22 / 0.70 / 0.98.
- **38, supported.** Sec. 5.4: fine-tuning a pretrained encoder with a learning rate 10 times smaller than the diffusion network's gave the best results overall. This was especially true for CLIP ViT-B/16, which reached 98% after 50 epochs.
- **39, supported, with two notes.** Sec. 6.1 and Table 6: end-to-end 0.95 success (IoU 0.80), R3M 0.80 (0.66), ImageNet 0.15 (0.24). 136 demonstrations (Table 3). Appendix: 2 of the 5 RealSense cameras were used, down-sampled to 320x240, with a 288x216 random crop. Authors' conclusion: end-to-end training is the most effective way. Note 1: I did not find the real end-effector described (the simulated Push-T uses a circular one), so "rod-like" is unverified. Note 2, which matters for the conclusion: "end-to-end" here is a ResNet-18 without any pretraining, with spatial softmax and GroupNorm (Sec. 3.2).
- **40, supported (still an inference).** The config sets training.freeze_encoder to True. It uses rgb_model resnet18 with IMAGENET1K_V1 weights (a comment says "or r3m"), resize_shape [224,224], crop_shape null, random_crop False, use_group_norm False, and lr 1e-4. Its workspace class does read freeze_encoder; when the flag is set, it puts the encoder in eval mode and switches off its gradients. So if this config produced Table 6, the pretrained variants were frozen, resized and not crop-augmented, while the end-to-end model used random crops. The paper itself does not say which.

### Mandlekar et al. (2021), robomimic, CoRL 2021 (venue not on the ar5iv page; not checked)
URL opened: https://ar5iv.labs.arxiv.org/html/2108.03298 (both halves)

- **41, supported.** App. D.1: ResNet-18 followed by spatial softmax. Sec. 4.3 and Fig. 2a: removing randomisation (random crops, e.g. 76x76 from 84x84) gives relative drops of 47% on Square and 35% on Transport. Image policies nearly match low-dimensional ones on proficient-human data (Table 3 against Table 1, e.g. Square 82.0 against 84.0; Tool Hang is higher with images). Proficient-human sets have 200 demonstrations, except Transport with 300 (App. B.1). The text I read does not say whether the ResNet-18 was pretrained.

### Zhao, Kumar, Levine, Finn (2023), ACT, RSS 2023 (venue not checked)
URLs opened: https://ar5iv.labs.arxiv.org/html/2304.13705 and https://raw.githubusercontent.com/tonyzhaozh/act/main/detr/models/backbone.py

- **42, supported, with notes.** ResNet-18 backbones on four 480x640 webcams (Sec. III, IV-C). 50 demonstrations per task, except Thread Velcro with 100 (Sec. V-B). About 5 hours on one 11 GB RTX 2080 Ti. The paper never says "jointly" in words, but Algorithm 1 updates all parameters with one loss. Code: the torchvision ResNet is built with pretrained=is_main_process() and norm_layer=FrozenBatchNorm2d. The line that would freeze early layers is commented out, so convolution weights train whenever lr_backbone > 0. The paper gives no ablation of these choices.

### Xie, Lee, Xiao, Finn (2023), arXiv 2307.03659 (no venue on the page)
URL opened: https://ar5iv.labs.arxiv.org/html/2307.03659

- **43, corrected.** App. A.3.1 and Sec. 5.2: the pretrained policies fit the training environments but leave a large gap. R3M fails on most factors except new distractors. CLIP is similar to the from-scratch CNN except on new object textures. The ResNet-50 versions were used (App. A.2.2). The correction: the paper never says whether R3M and CLIP were frozen or fine-tuned, so "frozen" should be dropped.
- **44, corrected.** Sec. 5.1 (real robot): 91.7% with no shift. Table texture and camera orientation (the paper's word in that sentence) caused the largest drops, to 52.8% and 45.8%. The mapping comes from word order. Other factors: background 88.9%, distractors 80.6%, lighting 83.3%. Fig. 6: on the robot, crop augmentation helped most for camera position and table texture, which is supported. The correction: "more training environments shrank the gap" (5 to 100 environments, maximum gap from 0.4 to below 0.1, Fig. 5a) is a simulation (Factor World) result, not a real-robot one.

### Lin, So, Mahalingam, Liu, Abbeel (2023), SpawnNet, arXiv 2307.03567 (no venue on the page)
URL opened: https://ar5iv.labs.arxiv.org/html/2307.03567

- **45, corrected.** Table 6 (held-out instances), Place Bag: SpawnNet+d 86.7 ± 6.4, SpawnNet 76.7 ± 10.8, LfS+aug+d 66.7 ± 7.6, R3M 20.0 ± 7.1. The paper defines "+d" as depth-conditioned. R3M is frozen and RGB-only. Table 3: SpawnNet 86.7 (the same value as the +d variant), frozen DINO 11.7 ± 4.4, full fine-tuning 8.3 ± 4.4, LoRA 28.3, last-layer fine-tuning 15.0, ProgressiveNet 51.7. Correct statement: 86.7 and 66.7 both use depth. The RGB-only SpawnNet scored 76.7, which is the relevant number for a project with one RGB camera. Demonstrations: 90 to 102 per task (App. C), which is supported. The DINO model is ViT-S/8 (App. B), also supported.

### Shang et al. (2024), Theia, CoRL 2024 (confirmed by the arXiv comments)
URLs opened: https://arxiv.org/html/2407.20179v2 and https://arxiv.org/abs/2407.20179

- **46, supported (location corrected).** Sec. 4.2: spatial tokens are consistently better than [CLS] for transformers, including MVP and VC-1. The evidence is in Fig. 4 (no numbers in the text), with all encoders frozen on CortexBench. Table 3c is Theia's own distillation ablation (spatial tokens 77.3 to 78.9 against [CLS] 50.0), not a test of MVP or VC-1.
- **47, supported.** App. D.3.1: frozen encoders gave 0% on toy-microwave cooking, while fine-tuned ones worked reasonably. In Table 2, Theia-B fine-tuned reached 58%, 52% and 40% on the sub-steps. There is no frozen entry in the table; the 0% appears only in the text.

### Tsagkas et al. (2025, v2 27 February 2026), arXiv 2511.10762 (preprint)
URL opened: https://arxiv.org/html/2511.10762v2

- **48, supported.** Sec. V-A: 14 frozen encoders (7 ResNets, 7 ViTs), 10 Meta-World tasks, 25 demonstrations. Sec. V-B: in-distribution success rose from 63.1% to 66.4%. The out-of-distribution gains are in Fig. 4 only. Attentive feature aggregation (AFA) is a cross-attention layer with a trainable query over per-patch tokens (ViTs) or channel embeddings (CNNs). The baseline uses the CLS token or average pooling. Table II (real KUKA planar pushing): out-of-distribution success rose from 0% to 100%, with 80% in distribution for both, over 10 in-distribution and 16 out-of-distribution rollouts in total. The demonstrations and camera for this task are not stated.

### Dong, Man, Tokmakov, Wang (2026), arXiv 2602.04880v1 (preprint, 4 February 2026)
URL opened: https://arxiv.org/html/2602.04880v1

- **49, supported.** Sec. 4: nine representations, frozen in the main experiments, with 50 demonstrations per task and a multi-task diffusion policy. The environment-specific winners are as stated. Table 1 average Pearson r: state probe 0.753, ImageNet accuracy -0.020, shape bias -0.069, segmentation -0.042. "Did not correlate" is true only on average. Per environment, the values swing widely: ImageNet accuracy is 0.587 on Meta-World, and shape bias is -0.905 on Meta-World but 0.633 on SimplerEnv (Google).

### Schafer et al. (2023; ALA Workshop 2025, confirmed on the page)
URL opened: https://ar5iv.labs.arxiv.org/html/2312.02312

- **50, supported.** DINOv2 was best in Minecraft (ViT-L/14, 32.00%) and in Minecraft Dungeons (ViT-B/14). In CS:GO, pretrained encoders were clearly worse than the best end-to-end ones (Table 3, Sec. 5.2). Sec. 5.1: 128x128 images are enough. Table 5: 280.48 minutes against 4.58 minutes, on one A100 80 GB according to the caption.

### Kumar, Raghunathan, Jones, Ma, Liang (2022), ICLR 2022 oral (confirmed by the arXiv comments)
URL opened: https://arxiv.org/abs/2202.10054

- **51, supported.** Abstract: fine-tuning is on average 2% better in distribution and 7% worse out of distribution than linear probing. LP-FT is 1% better in distribution and 10% better out of distribution than full fine-tuning. 10 datasets.

### Caron et al. (2021), DINO
URL opened: https://arxiv.org/abs/2104.14294

- **52, supported.** Abstract: self-supervised ViT features hold explicit segmentation information, less visible in supervised ViTs or convnets. k-NN reaches 78.3% top-1 with a small ViT (and linear evaluation 80.1% with ViT-Base). The arXiv comments only say 21 pages, so ICCV is not shown there.

### Oquab et al. (2023/2024), DINOv2
URLs opened: https://arxiv.org/html/2304.07193v2 (both halves) and https://github.com/facebookresearch/dinov2

- **53, supported.** Tables 10 and 11: frozen ViT-S/14 with linear heads beats OpenCLIP ViT-G/14 on every segmentation and depth number shown (e.g. ADE20k linear 44.3 against 39.3; NYUd linear RMSE 0.449 against 0.541). Table 17: distilled ViT-S/14 has 384 dimensions, 6 heads and 12 blocks. README: 21M parameters; DINOv2 code and weights under Apache 2.0, except the Cell-DINO and XRay-DINO parts, which use other licences. The page links an OpenReview review but does not name TMLR.

### torchvision model cards (accessed 9 November 2026)
URLs opened: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html, .../torchvision.models.vit_b_16.html, .../torchvision.models.resnet50.html

- **54, supported (one sentence unsourced but correct).** ResNet-18: 11,689,512 parameters, 1.81 GFLOPS, 69.758% top-1. ViT-B/16: 86,567,656 parameters, 17.56 GFLOPS. The cards do not say how GFLOPS are counted. That they are multiply-adds agrees with my own count (ViT-B/16 comes out at 17.56 G multiply-adds) and with ResNet-50's 4.09.

### Own calculations by the researcher, re-done by me

- **55, supported (arithmetic re-done).** Per block: 12·N·D² + 2·N²·D, plus a patch embedding of 224·224·3·D. LayerNorm, softmax, GELU and the head are ignored. Results: ViT-S/16 (N = 197, D = 384) 4.60 GMACs. DINOv2 ViT-S/14 at 224 px (N = 257) 6.12. ViT-L/16 (D = 1024, 24 blocks) 61.55. ViT-B/16 17.56, which equals torchvision.
- **56, supported as arithmetic; still a guess.** The hidden assumption is that one training step costs about 3 times a forward pass. At 224 px: 20 epochs × 18,500 = 370,000 image passes × 5.43 G = 2.0 × 10^6 GMAC, which takes 2.8 h at 200 GMAC/s and 11.2 h at 50. At 160x120: 1.1 to 4.3 h. Frozen features once: 18,500 × 1.81 G, which takes 2.8 to 11.2 min. Left out: JPEG decoding, augmentation, data loading and laptop thermal throttling. The 50 to 200 GMAC/s range is unmeasured, so it must be timed.

### Radford et al. (2021), CLIP, ICML 2021, PMLR 139:8748-8763 (confirmed)
URL opened: https://proceedings.mlr.press/v139/radford21a.html

- **57, supported.** Abstract: trained from scratch on 400 million (image, text) pairs collected from the internet.

## Challenge to the conclusion

### The "Evidence" paragraph: mostly right, three slips

1. Hansen et al.'s competitive from-scratch model is a shallow 4- to 6-layer ConvNet, not a ResNet-18.
2. Chi et al.'s real Push-T winner (95%) is a ResNet-18 with no pretraining (GroupNorm, spatial softmax, random crop). The pretrained variants (80%, 15%) were probably frozen, resized to 224 and not crop-augmented (claim 40). That comparison mixes pretraining with freezing and with augmentation, so it supports end-to-end training, not ImageNet initialisation.
3. "Frozen DINO ViT-S was the most robust" goes further than Burns et al.'s text (claim 32). Their test is also frozen behaviour cloning with 10 demonstrations in simulation.

Two more caveats:
- Dasari et al.'s "ImageNet" is self-supervised MAE (ViT-B) or SimCLR (ResNet-18) on ImageNet images, not supervised torchvision weights.
- "Robot-specific models shine mainly in frozen simulation tests" is fair, but R3M (real Franka, 56% against CLIP's 24%), MVP (real xArm) and VC-1 (real Franka, 70% frozen) also won real-robot tests. Those wins were against weak frozen baselines.

### Does the evidence support "ImageNet ResNet-18, fine-tuned" as the safest choice?

As a sensible default, yes. As the choice the evidence proves, no.

No opened study compares supervised ImageNet ResNet-18 fine-tuning with a ResNet-18 or small CNN trained from scratch at about 200 demonstrations and finds a clear winner:
- Chi Table 5: ResNet-18 scratch 0.94 against fine-tuned 0.92, a tie. The best result was a fine-tuned CLIP ViT-B/16.
- Hansen Table 5: fine-tuned ImageNet MoCo with augmentation (65.0) only tied the from-scratch model with augmentation (63.1). Fine-tuned R3M with augmentation (80.5) clearly beat it.
- ACT's ImageNet ResNet-18 with frozen BatchNorm is a design choice, not an ablation.

In distribution, end-to-end training with augmentation is what the evidence supports, whatever the initialisation. The case for ImageNet initialisation rests on robustness. That support is indirect and mixed: Hansen's DMC-GB table found from-scratch training with augmentation more robust than frozen pretrained models.

### Fit to this project's situation

- **Unseen tracks.** No opened study tests new path shapes. The shifts tested are appearance, object instances and camera pose. With only six training sheets, the bigger risks are probably:
  - shortcut learning from sheet-specific global cues (sheet edges, table, position on the sheet);
  - too little coverage of local situations (curvature range, the crossing, dash gaps).

  The backbone choice is unlikely to be the main risk. Crop or shift augmentation and pooling that attends to task-relevant spatial cues (Theia, Tsagkas) address this more directly than pretraining does.
- **Layer choice.** The scene is a black line and a red puck on white paper. Parisi found layer-3 features best for fine control. PIE-G found layer 2 of ResNet-18 generalised best (RL). Cutting ResNet-18 after layer 3 keeps the spatial map, saves about a quarter of the compute and about 70% of the parameters (my own count). It is worth one comparison.
- **BatchNorm.** Frozen BatchNorm is copied from ACT without evidence. Diffusion Policy used GroupNorm instead. PIE-G found that updating the BatchNorm statistics helped generalisation. Treat it as a setting to test, with frozen as the safe start for small CPU batches.
- **CPU budget.** Fine-tuning at 224 px costs about 3 to 11 h per run by the researcher's own estimate. That makes six leave-one-sheet-out folds 17 to 67 h per configuration. The evidence says lower resolution is enough: robomimic used 84x84 and 120x120 images with ResNet-18, and Schafer found 128x128 enough. A practical order:
  1. Screen with cached frozen features (minutes each).
  2. Fine-tune only the winner at about 128x96 or 160x120, in the LP-FT style (Kumar).

  Note that cached features rule out pixel augmentation, unless several augmented copies are cached.
- **Model selection.** "Choose by leave-one-sheet-out" is right, but offline scores can mislead. In robomimic, the checkpoint with the lowest validation loss scored 7.3% where the best scored 84.0% (Square, low-dimensional, Table 29). Dasari found simulated success barely predicted real success. Final choices therefore need closed-loop runs on held-out sheets. At about 1.2 s per step, each run takes only minutes.
- **Noise.** Most real-robot results come from one task with 10 to 20 trials, with standard errors of 7 to 15 points. Gaps smaller than that should not drive decisions.
- **Forearm roll.** The forearm roll rotates the picture and carries state that the model needs. Rotation augmentation would need label correction (as in PilotNet's shifted and rotated views). No backbone study covers this.

### Suggested amendment

Keep the ImageNet ResNet-18 as the main candidate, at reduced resolution. Add two candidates to the same leave-one-sheet-out test:
- a Hansen-style small CNN trained from scratch with random shift (cheap and evidence-backed);
- a ResNet-18 truncated at layer 3.

Test BatchNorm handling. Make the final choice from closed-loop runs on held-out sheets. Skipping ViT-L is well supported: it costs about 34 times ResNet-18, it collapses when fine-tuned, and R3M (ResNet-50) beat it on frozen behaviour cloning.

## Evidence the researcher missed

1. **Chi et al. (2023), Sec. 3.2 (checked, https://arxiv.org/html/2303.04137).** The default and best real encoder is a ResNet-18 without pretraining, with spatial softmax pooling and GroupNorm instead of BatchNorm. This changes how the Push-T numbers should be read.
2. **Hansen et al. (2023), Sec. 2 and App. B.1 (checked).** The competitive from-scratch model is a shallow 4- to 6-layer ConvNet with random shift (padding 12). That is a cheap CPU baseline.
3. **Mandlekar et al. (2021), appendix and Table 29 (checked, https://ar5iv.labs.arxiv.org/html/2108.03298).**
   - Selecting the checkpoint with the lowest validation loss can be far worse than the best checkpoint: Square low-dimensional 7.3% against 84.0%.
   - Images were 84x84 in simulation and 120x120 on the real robot, with random crops of 76x76 and 108x108.
4. **Yuan, Xue, Yuan, Wang, Wu, Gao, Xu (2022), PIE-G, NeurIPS 2022 (checked on https://ar5iv.labs.arxiv.org/html/2212.08860; the NeurIPS page refused the connection).**
   - The encoder is frozen layer 2 of an ImageNet ResNet-18 with its BatchNorm statistics kept updating. Layer 2 beat layers 1, 3 and 4.
   - Fine-tuning hurt.
   - The average gain on DMC-GB video-hard was 55%.
   - This is RL, not behaviour cloning.
5. **Shafiullah et al. (2023), On Bringing Robots Home (Dobb-E), arXiv 2311.16098 (checked, https://ar5iv.labs.arxiv.org/html/2311.16098; no venue on the page).** This is the closest match to the project's set-up:
   - an iPhone camera mounted on the robot arm;
   - a ResNet-34 pretrained with MoCo-v3 on the team's own home data, then fully fine-tuned (lr 3e-5, 50 epochs);
   - about 24 demonstrations per task.

   It beat MVP, R3M, VC-1 and ImageNet-1K models (Sec. 3.4.1, Fig. 21; at least 23% by their text; I could not read the bars). This is relevant for a single arm-mounted camera and for self-supervised pretraining on the team's own pictures.
6. **Bojarski et al. (2016), End to End Learning for Self-Driving Cars, arXiv 1604.07316 (checked, https://ar5iv.labs.arxiv.org/html/1604.07316).**
   - One front camera.
   - A CNN with about 250 thousand parameters.
   - About 72 hours of driving.
   - Artificially shifted and rotated views with corrected steering labels, which teach recovery.
   - About 98% autonomy.

   This is the nearest analogue to following unseen tracks. The paper does not say whether the CNN was pretrained.
7. **Majumdar et al. (2023), Table 2 detail (checked).** On the few-demonstration behaviour-cloning benchmarks, frozen R3M (ResNet-50) beat VC-1 (ViT-L) by 7 to 14 points. VC-1's best mean comes from navigation tasks.
8. **Kumar et al. (2022), LP-FT (checked).** It was cited but not used in the judgement. Training the head on frozen features first and then fine-tuning everything is a cheap recipe that fits the "frozen first, then fine-tune" plan.
9. **Not opened, worth reading:**
   - Pari et al. (2021), VINN: self-supervised BYOL on the team's own demonstrations plus nearest neighbours.
   - Hsu et al. (2022), Vision-Based Manipulators Need to Also See from Their Hands: hand-centric views and generalisation.
   - Ross, Gordon, Bagnell (2011), DAgger: compounding errors, which matters because finishing counts most.
   - Kadian et al. (2020): the definition of SRCC used by Silwal et al.
   - Laskin et al. (2020), RAD, and Kostrikov et al. (2020), DrQ: the random shift augmentation itself.
