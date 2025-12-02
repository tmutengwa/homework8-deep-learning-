# Machine Learning Zoomcamp - Homework 8 Solution

Deep Learning with PyTorch - Hair Classification (Straight vs Curly)

## Files Included

1. **`homework8_solution.ipynb`** - Jupyter notebook (recommended)
4. **`requirements.txt`** - Required Python packages
5. **`README.md`** - This file

## Setup

### Local Jupyter Notebook

1. Install dependencies:
   ```bash
   pip install torch torchvision numpy pillow
   ```

2. Download the data (if not already downloaded):
   ```bash
   wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
   unzip data.zip
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook homework8_solution.ipynb
   ```

```bash
pip install torch torchvision numpy pillow
```

## Model Architecture

```
Input: (3, 200, 200)
    ↓
Conv2d(3 → 32, kernel=3×3, padding=0, stride=1)
    ↓
ReLU
    ↓
MaxPool2d(2×2)
    ↓
Flatten → (313,632)
    ↓
Linear(313,632 → 64)
    ↓
ReLU
    ↓
Linear(64 → 1)
    ↓
Output (Logit for BCEWithLogitsLoss)
```

## Training Details

### Phase 1: Without Augmentation (10 epochs)
- Batch size: 20
- Optimizer: SGD (lr=0.002, momentum=0.8)
- Loss: BCEWithLogitsLoss
- Transformations: Resize, ToTensor, Normalize

### Phase 2: With Augmentation (10 more epochs)
- Same settings as Phase 1, plus:
- RandomRotation(50°)
- RandomResizedCrop(200, scale=(0.9, 1.0))
- RandomHorizontalFlip()


### CUDA/GPU
The code will automatically use GPU if available, otherwise CPU.
Training on CPU will be slower but still works.

## Notes

- Random seeds are set for reproducibility (SEED=42)
- Results may still vary slightly between runs
- Training takes ~5-10 minutes on CPU, ~1-2 minutes on GPU
- The data directory should be at `./data/train` and `./data/test`

## References

- ML Zoomcamp Course: https://github.com/DataTalksClub/machine-learning-zoomcamp
- PyTorch Documentation: https://pytorch.org/docs/
# homework8-deep-learning-
