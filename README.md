# 🟢 Shape Classifier (PyTorch)

This project demonstrates the training and evaluation of a simple CNN-based vision model to classify geometric shapes — **Circle**, **Square**, and **Triangle** — using synthetic data generated with the Python PIL library.

---

## 📌 Task Objective

Develop a basic shape classifier under different synthetic conditions:
- Fixed Length, Fixed Rotation ✅
- Fixed Length, Random Rotation ⏳
- Random Length, Fixed Rotation ⏳
- Random Length, Random Rotation ⏳

---

## 🧪 Current Status
| Condition Type | Implemented | Accuracy |
|----------------|-------------|----------|
| Fixed Length + Fixed Rotation | ✅ Done | 100% (Train / Val / Test) |

---

## 🧠 Model Summary

A simple CNN with 3 convolutional layers and 2 fully connected layers is used. It efficiently classifies synthetic shapes under the fixed size + rotation condition with 100% accuracy.

```python
Conv2d → ReLU → MaxPool  
Conv2d → ReLU → MaxPool  
Conv2d → ReLU → MaxPool  
Flatten → FC → ReLU → FC (3 outputs)
```
---

## 📂 Folder Structure

ShapeClassifierPyTorch/
├── Shape_Classifier.ipynb         # 📓 Main notebook for training and evaluation
├── requirements.txt               # 📦 Dependencies for pip install
├── README.md                      # 📘 Project overview and instructions
├── images/                        # 📷 Visualizations for README
│   └── sample_preds.png
├── data/                          # 🖼️ Generated synthetic datasets
│   └── fixed_length_fixed_rotation/
│       ├── circle/
│       ├── square/
│       └── triangle/
└── saved_models/                  # 💾 (Optional) model checkpoints

## 🖼️ Sample Dataset Visualization

### 📊 Example of Synthetic Shapes (Fixed Length, Fixed Rotation)

This figure shows a few randomly generated shapes from the dataset used to train the classifier:

<img src="images/sample_dataset.png" width="600"/>

---

## 📉 Training Log Summary

The model was trained for 15 epochs using the Fixed Length + Fixed Rotation dataset. It converged very quickly due to the simplicity and clarity of the synthetic data.

**Device:** CUDA (GPU)  
**Loss Function:** CrossEntropyLoss  
**Optimizer:** Adam (lr = 0.001)  
**Batch Size:** 32  
**Epochs:** 15

**Training Summary:**

Using device: cuda
[Epoch 1/15] Train Loss: 0.3529 | Val Loss: 0.0000 | Val Acc: 100.00%
[Epoch 2/15] Train Loss: 0.0000 | Val Loss: 0.0000 | Val Acc: 100.00%
...
[Epoch 15/15] Train Loss: 0.0000 | Val Loss: 0.0000 | Val Acc: 100.00%


The model reached **100% accuracy on both training and validation** within 2 epochs and maintained it across all 15, showing perfect generalization under this condition.
