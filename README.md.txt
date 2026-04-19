#  Precision Farming: AI-Powered Plant Disease Detection

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

##  Project Overview
Plant diseases pose a significant threat to global food security, reducing crop yields by up to 30-40% annually. Traditional manual inspection is time-consuming, prone to human error, and lacks scalability. 

This project implements an automated **Computer Vision** system using deep learning to classify plant leaf images into healthy or diseased categories, enabling early intervention and precision agricultural monitoring.

##  Dataset
The model is trained on the **PlantVillage Dataset**, specifically targeting the Tomato subset, which includes:
- **15 distinct health classes** (e.g., Early Blight, Late Blight, Target Spot, Healthy).
- Over **20,000 augmented images** simulating real-world field conditions (lighting variations, angles).

##  Model Architecture
A custom Sequential **Convolutional Neural Network (CNN)** was built from scratch to automatically extract spatial hierarchies of features without the need for manual feature engineering.

**Pipeline Highlights:**
1. **Data Augmentation:** Real-time transformations (`rotation`, `width_shift`, `horizontal_flip`) to prevent overfitting.
2. **Feature Extraction Base:** Multiple `Conv2D` (32 & 64 filters) + `MaxPooling2D` blocks.
3. **Classification Head:** `Flatten` layer followed by a `Dense` layer (128 neurons, ReLU) and a Softmax output layer.
4. **Optimization:** Adam optimizer with Categorical Crossentropy.
5. **Callbacks:** Implemented `EarlyStopping` and `ModelCheckpoint` for optimal convergence.

##  Results & Performance
The model successfully converged, demonstrating robust classification capabilities across all 15 categories.

* **Final Validation Accuracy:** **92.55%**
* **Training vs. Validation Loss:** Handled efficiently with early stopping and data augmentation to prevent over-fitting.

![Training Metrics](assets/training_metrics.png)

##  Author
**Mostafa Mohamed** | AI Engineer 
