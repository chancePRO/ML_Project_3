# 📘 HOG vs Not-HOG Logo Classification (CNN Project)

## 👥 Group # — CNN Project

This project implements a Convolutional Neural Network (CNN) to classify images as the **official HOG logo** or **not the official HOG logo**.

The workflow:

1. Load and preprocess the dataset  
2. Build CNN architectures  
3. Train and evaluate multiple models  
4. Tune architectures and parameters  
5. Select the best-performing model  
6. Save the final model  
7. Reflect on design choices and future improvements  

---

## 1. Dataset

The dataset directory is expected to look like:

```text
hog_vs_not/
    ├── hog/
    └── not_hog/
```

Each subfolder contains images for that class.  
We load the dataset using `torchvision.datasets.ImageFolder`.

We use an **80/20 random split** for training and validation using `torch.utils.data.random_split`.

---

## 2. Model: HogCNN

We use a configurable CNN called `HogCNN`. Key ideas:

- 3 convolutional blocks
- Each block: Conv2d → ReLU → Pool (MaxPool2d or AvgPool2d)
- Final layers: Flatten → Linear → ReLU → Dropout → Linear → Output logits

The architecture can be controlled by:

- `channels` (e.g., [16, 32, 64] or [32, 64, 128])
- `pool_types` (e.g., ["max", "max", "max"] or ["max", "avg", "max"])
- `conv_kernels` (e.g., [3, 3, 3] or [5, 3, 3])

This allows us to try different architectures without rewriting the model.

---

## 3. Architecture & Hyperparameter Search

To satisfy the requirement of developing multiple candidate solutions, we define a list of architecture configs:

```python
arch_configs = [
    {"name": "small_max",         "channels": [16, 32, 64], "pool_types": ["max","max","max"], "conv_kernels": [3,3,3]},
    {"name": "medium_max",        "channels": [32, 64,128], "pool_types": ["max","max","max"], "conv_kernels": [3,3,3]},
    {"name": "medium_mix",        "channels": [32, 64,128], "pool_types": ["max","avg","max"], "conv_kernels": [3,3,3]},
    {"name": "large_kernels_avg", "channels": [32, 64,128], "pool_types": ["avg","avg","avg"], "conv_kernels": [5,3,3]},
]
```

For each configuration we:

1. Build `HogCNN` with those parameters  
2. Train for a fixed number of epochs using the training set  
3. Evaluate on the validation set each epoch  
4. Track the best validation accuracy for that architecture  

We store results in a list of dictionaries and print a summary of each architecture’s performance.

---

## 4. Training & Evaluation

Training uses:

- **Loss**: `nn.CrossEntropyLoss`
- **Optimizer**: `torch.optim.Adam`
- **Metrics**:
  - Training loss & accuracy per epoch
  - Validation loss & accuracy per epoch

After all architectures are trained, we identify the **best model** by highest validation accuracy:

- Save its `state_dict`
- Save its architecture configuration
- Restore that model for plotting and saving

Box 7 in the notebook plots:

- Training vs validation loss
- Training vs validation accuracy

for the **best-performing architecture**.

---

## 5. Saving the Final Model

The final selected model is saved with the required naming convention:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "best_arch_config": best_arch_config,
}, "Group_#_CNN_FullModel.ph")
```

Replace `#` with your actual group number.

---

## 6. Reflection Points (for Report / Writeup)

These are not code, but the notebook supports them:

1. **Dataset creation & split**  
   - Images are organized into `hog/` and `not_hog/` folders.  
   - An 80/20 random split is used for train/validation.

2. **Final architecture choice**  
   - Multiple architectures are tested.  
   - The final architecture is chosen based on best validation accuracy.  
   - You can discuss why the winning configuration makes sense (capacity, pooling choice, kernels).

3. **Monitoring & mitigating overfitting**  
   - Overfitting is monitored with train vs validation curves.  
   - Mitigation: architecture selection based on validation performance, dropout in the classifier, limited epochs, and potential data augmentation.

4. **Future improvements**  
   - Collect more images and more diverse backgrounds.  
   - Use stronger augmentations.  
   - Add BatchNorm and possibly deeper models.  
   - Use a separate test set for final evaluation.  
   - Experiment with learning rate schedules and more hyperparameters.

---

## 7. How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your dataset is in:

   ```text
   hog_vs_not/
       hog/
       not_hog/
   ```

3. Open the notebook (e.g., `Code04.ipynb`) and run cells in order.

4. After training:
   - The script prints the best architecture and its validation accuracy.
   - Loss and accuracy curves are displayed for the best model.
   - The final model is saved as `Group_#_CNN_FullModel.ph`.

---

## 8. Requirements

See `requirements.txt` for specific package versions. At a high level, this project uses:

- Python 3.x  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- Pillow  
