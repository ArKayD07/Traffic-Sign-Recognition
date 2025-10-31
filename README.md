# Traffic Sign Recognition Extended Essay

This project provides a full pipeline for traffic sign recognition using deep learning, including data generation, augmentation, annotation conversion, training, and evaluation.

## Folder Structure
- `main.py` — Orchestrates the full pipeline; run this to execute all steps
- `generate_synthetic.py` — Generates synthetic traffic sign images
- `convert_annotations.py` — Converts and splits annotations to various formats
- `augmentation.py` — Augments training images
- `train_and_eval.py` — Trains a ResNet-18 classifier
- `evaluate_and_sanity_check.py` — Evaluates trained models and produces metrics
- `evaluate_and_save_named_preds.py` — Evaluates and saves predictions for named models
- `compare_models.py` — Compares multiple models (if present)
- `dataset/`, `augmented_dataset/`, `converted_dataset/`, `results/` — Data and output folders

## Prerequisites
- Python 3.8+
- Recommended: virtual environment (venv or conda)
- Required Python packages:
  - torch
  - torchvision
  - pillow
  - albumentations
  - opencv-python
  - tqdm
  - pandas
  - scikit-learn
  - matplotlib

Install all dependencies:
```powershell
pip install torch torchvision pillow albumentations opencv-python tqdm pandas scikit-learn matplotlib
```

## How to Run the Full Pipeline
1. Open a terminal in the `Traffic Sign Recognition Extended Essay` folder.
2. Run the orchestrator:
```powershell
python main.py --all
```
This will execute all main steps in order (data generation, conversion, augmentation, training, evaluation).

### Run Only Specific Steps
```powershell
python main.py --steps generate_synthetic,convert_annotations,train_and_eval
```

### List Available Scripts
```powershell
python main.py --list
```

### Use a Specific Python Interpreter
```powershell
python main.py --all --python "C:\Path\To\venv\Scripts\python.exe"
```

### Continue on Error
```powershell
python main.py --all --continue_on_error
```

### Dry Run (show commands, don't execute)
```powershell
python main.py --all --dry_run
```

## Notes
- Each script can be run individually if needed; see the top of each `.py` file for CLI options.
- Output artifacts (models, metrics, augmented data) will be saved in the appropriate folders.
- For large datasets or GPU training, adjust script arguments in `main.py`'s `DEFAULT_ARGS` mapping.
- If you encounter missing packages, install them using `pip` as shown above.

## Troubleshooting
- If you see errors about missing files or folders, check that your dataset and annotation paths are correct.
- For CUDA/GPU issues, ensure your PyTorch installation matches your hardware.
- For help with a specific script, run it with `--help` (e.g., `python train_and_eval.py --help`).

---

For questions or further customization, edit `main.py` or any script as needed.