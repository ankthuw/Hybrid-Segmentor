
# Hybrid-Segmentor – Crack Segmentation on CrackVision12K

This repository implements the Hybrid-Segmentor model for automated fine-grained crack segmentation in civil infrastructure images. The model is composed of a **dual-path encoder** (CNN + Transformer) and a simplified decoder, enabling it to segment both prominent and thin cracks with high precision.

This work includes:

- Full training pipeline (with `trainer.py`)
- Evaluation on validation and test sets (`test.py`)
- Independent training of **CNN-only** and **Transformer-only** branches for ablation analysis

---

## Dataset: CrackVision12K

We use a refined crack segmentation dataset consisting of 12,000 256×256 RGB images with binary ground truth masks. The dataset is pre-split as follows:

```
dataset/
├── train/
│   ├── IMG/
│   └── GT/
├── val/
│   ├── IMG/
│   └── GT/
└── test/
    ├── IMG/
    └── GT/
```

---

## Configuration

All hyperparameters and dataset paths can be modified in `config.py`.

Example values:

```python
TRAIN_IMG_DIR = "/kaggle/working/dataset/train/IMG"
TRAIN_MASK_DIR = "/kaggle/working/dataset/train/GT"
VAL_IMG_DIR = "/kaggle/working/dataset/val/IMG"
VAL_MASK_DIR = "/kaggle/working/dataset/val/GT"
TEST_IMG_DIR = "/kaggle/working/dataset/test/IMG"
TEST_MASK_DIR = "/kaggle/working/dataset/test/GT"
```

---

## Training

```bash
python trainer.py
```

- The script automatically selects GPU if available.
- Model checkpoints are saved under `checkpoints/v7_BCEDICE0_2_final/`.
- Training uses Dice+BCE Loss and early stopping.
- Model is tested at the best-performing checkpoint (lowest val loss).

---

## Testing

To evaluate the model on the test set:

```bash
python test.py
```

- Set the checkpoint path and output directory in `test.py`.
- Output includes per-image predictions and overall metrics (IoU, Dice, Precision, Recall).

<!-- --- -->

<!-- ## Ablation Experiments

To isolate and evaluate the impact of each encoder branch, run:

```bash
# CNN-only mode
MODEL_TYPE = "cnn_only"

# Transformer-only mode
MODEL_TYPE = "transformer_only"

# Full hybrid (default)
MODEL_TYPE = "hybrid"
```

This is controlled via internal switches in `model.py` or passed via config. -->

<!-- --- -->

<!-- ## Results

We evaluated three configurations:

| Model              | Accuracy | Precision | Recall | Dice   | IoU   |
|-------------------|----------|-----------|--------|--------|--------|
| CNN-only          | 0.9627   | 0.7167    | 0.7480 | 0.7320 | 0.5773 |
| Transformer-only  | 0.9611   | 0.7188    | 0.7033 | 0.7110 | 0.5516 |
| **Hybrid (Full)** | **0.9689** | **0.8058** | **0.7280** | **0.7585** | **0.6177** |

These results confirm that the hybrid model outperforms both branches individually, especially in segmentation accuracy and precision.

--- -->

<!-- ## Qualitative Results

Examples of prediction results on diverse crack patterns:

- [x] Large continuous cracks (e.g. Hình 2,3)
- [x] Fine cracks in noisy background (e.g. Hình 1)
- [x] Multi-branching patterns and weak signals

![](figures/sample_preds.png) -->

---

## Citation

If you use this codebase or reproduce our experiments, please cite the original paper:

```
@article{goo2024hybridsegmentor,
  title={Hybrid-Segmentor: A Hybrid Approach to Automated Fine-Grained Crack Segmentation in Civil Infrastructure},
  author={Goo, June Moh and Milidonis, Xenios and Artusi, Alessandro and Boehm, Jan and Ciliberto, Carlo},
  journal={arXiv preprint arXiv:2409.02866},
  year={2024}
}
```

---

## Contact

- Project Owner: [Anh Thư - HCMUS]
- Notebooks:
  - [Hybrid-Segmentor Full](https://www.kaggle.com/code/thblanh/computervision-project)
  - [CNN Branch](https://www.kaggle.com/code/nhuttrang/cnn-branch)
  - [Transformer Branch](https://www.kaggle.com/code/nhuttrang/transformer-branch)
