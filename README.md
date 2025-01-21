# SignLanguageTranslator

## Introduction
This project adapts various CNN-based architectures to identify letters in three sign language fingerspelling datasets:

- **LIS:** Italian Sign Language
- **ASL:** American Sign Language
- **BSL:** British Sign Language

We employed five pre-trained networks and fine-tuned them to maximize accuracy. Key strategies included:

- Experimenting with batch sizes (16, 32, 64) and epochs (10, 15, 20)
- Incorporating batch normalization and dropout for better generalization
- Balancing datasets to ensure uniformity within and across datasets

---

## Results Storage
Results are stored as images in model-specific branches, including:

1. **Confusion Matrices:** For each dataset and batch size/epoch combination (16-10, 32-15, 64-20).
2. **Summary Tables:** Overview of performance metrics like accuracy, precision, recall, and F1 scores.

---

## Datasets
The datasets were sourced from Kaggle and include only the folders with letters (not numbers):

- **ASL Dataset:**
  - 26 classes (A-Z), 3,000 images per class.
- **LIS Dataset:**
  - 22 classes, with 240-330 images per class (unbalanced).
- **BSL Dataset:**
  - 23 classes, 1,000 images per class, with noisy backgrounds.

### Dataset Balancing
We balanced the datasets using custom scripts, resulting in six versions:

- ASL(1000): Reduced to 1,000 images/class
- ASL(300): Reduced to 300 images/class
- BSL(1000): Original dataset
- BSL(300): Reduced to 300 images/class
- LIS(300): Augmented to 300 images/class
- LIS(1000): Augmented to 1,000 images/class

### Combined Dataset
A combined dataset (ASL(300), BSL(300), LIS(300)) was created to enable cross-dataset learning.

---

## CNN Architectures

### ResNet50
A deep network with residual connections to address vanishing gradients. Key observations:

- **Performance:** Achieved 89%-97% accuracy across datasets.
- **Challenges:** Difficulties with similar gestures (e.g., M/N for LIS, R/U/V for ASL).

### EfficientNetV2
A scalable network balancing width, depth, and resolution. Notable results:

- High performance across all datasets with minimal variance.
- Minor misclassifications like F/B (LIS) and M/N (BSL).

### MobileNetV2
An efficient architecture using inverted residuals. Findings:

- Acceptable performance but struggled with certain letters (e.g., I, E, Y for ASL).
- Less robust on the combined dataset.

### VGG16 & VGG19
Deep networks with simple architectures:

- **VGG19:** Performed better on individual datasets (e.g., LIS(300), BSL(300)).
- **VGG16:** Unexpectedly outperformed VGG19 on the combined dataset.

---

## Final Results

### Summary:
- **Best Overall Model:** ResNet50 for accuracy and robustness.
- **Dataset-Specific Performance:**
  - **LIS(300):** VGG19 (98%)
  - **BSL(300):** ResNet50 (97%)
  - **ASL(300):** ResNet50 (96%)
  - **Combined Dataset:** ResNet50 maintained robustness; others showed lower performance.

---

## Recommendations
- **ResNet50:** For high accuracy across diverse datasets.
- **VGG19:** For datasets requiring deeper feature learning.
- **EfficientNetV2:** Balanced choice for mixed datasets.

---

## References
1. [ASL Dataset - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. [LIS Dataset - Kaggle](https://www.kaggle.com/datasets/nicholasnisopoli/lisdataset)
3. [BSL Dataset - Kaggle](https://www.kaggle.com/datasets/erentatepe/bsl-numbers-and-alphabet-hand-position-for-mediapipe)
4. He, K. et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
5. Simonyan, K., "Very Deep Convolutional Networks for Large-Scale Image Recognition," 2014.
6. Tan, M., Le, Q.V., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," 2019.
