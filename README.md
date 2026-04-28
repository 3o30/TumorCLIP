## TumorCLIP

This repository is the paper code release for TumorCLIP: Lightweight Vision–Language Fusion for Explainable MRI-Based Brain Tumor Classification

doi: https://doi.org/10.64898/2026.03.11.26348155

Reproducibility is provided via **three Jupyter notebooks**; core modules live under `src/`.

## Project structure

```text
AI4BrainTumorDiagnosis-final/
├── Enhanced_Single_Modal_Training.ipynb      # Train single-modal baselines from scratch and save best weights
├── CLIP_Fusion_Model_Training.ipynb          # Train CLIP fusion model using single-modal weights
├── Model_Evaluation_Visualization.ipynb      # Evaluation & visualization (confusion matrix, ROC, etc.)
├── requirements.txt
└── src/
    ├── config/                               # Constants (class names, prompts, etc.)
    ├── core/                                 # Utilities & logging (JSON saving, timing, etc.)
    ├── data/                                 # Dataset/transform/dataloader factory
    ├── docs/                                 # Formula-to-code mapping notes
    ├── models/                               # Losses, DenseNet variants, single-modal model factory
    ├── training/                             # Single-modal trainer and enhanced trainer
    └── visualization/                        # Plotting and evaluation utilities
```

## Environment and installation

- **Recommended Python**: 3.9+
- **Install dependencies**:

```bash
pip install -r requirements.txt
```

- **Extra dependency for CLIP fusion**: `CLIP_Fusion_Model_Training.ipynb` imports `open_clip`. If it is not available in your environment, install `open-clip-torch`:

```bash
pip install open-clip-torch
```

## Dataset location and layout

Place the dataset under the project root at `data/` (same level as `src/`), following the `torchvision.datasets.ImageFolder` layout:

```text
data/
├── train/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── NORMAL/
│   ├── Neurocitoma/
│   ├── Outros Tipos de Lesões/
│   └── Schwannoma/
└── test/
    ├── Glioma/
    ├── Meningioma/
    ├── NORMAL/
    ├── Neurocitoma/
    ├── Outros Tipos de Lesões/
    └── Schwannoma/
```

Notes:
- This repository **does not include the dataset**. Please obtain it from the paper/data source and place it in the directory structure above.
- The notebooks default to `data/train` and `data/test`. If you store data elsewhere, update `DATA_TRAIN_PATH/DATA_TEST_PATH` or `train_dir/test_dir` in the corresponding notebook configuration cells.

## Reproducibility (recommended order)

### 1) Single-modal training (produce the best DenseNet weights)

Run `Enhanced_Single_Modal_Training.ipynb`.

Key outputs (examples):
- `results/best_models/DenseNet121_Adam_lr0.0001_best.pth`
- `results/training_logs/enhanced_single_modal_results.json`

### 2) CLIP fusion training (depends on single-modal best weights)

Run `CLIP_Fusion_Model_Training.ipynb`.

By default, the notebook loads the single-modal weights from:
- `results/best_models/DenseNet121_Adam_lr0.0001_best.pth`

### 3) Evaluation and visualization

Run `Model_Evaluation_Visualization.ipynb` to evaluate saved models and generate visualizations.

## Output directory convention

The following directories will be created automatically (if missing):
- `results/`: experiment outputs and visualizations
  - `results/best_models/`: best checkpoints (`.pth`)
  - `results/training_logs/`: training logs and summary JSON
  - `results/plots/` or `results/visualizations/`: figures (exact subfolder depends on the notebook/module)


## Notes

- **GPU recommended**: training and fusion are strongly recommended to run on GPU; CPU will work but will be slow.


## Citation

If you use this repository in your work, please cite the corresponding paper (we recommend adding BibTeX or `CITATION.cff` when publishing).
