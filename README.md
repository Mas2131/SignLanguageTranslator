# ReadMe

## Introduction
This project focuses on adapting various CNN-based architectures to identify the letters in three different sign language fingerspelling datasets:

- **LIS:** Italian Sign Language;
- **ASL:** American Sign Language;
- **BSL:** British Sign Language;

To achieve this, we employed five pre-trained networks and applied fine-tuning techniques to maximize accuracy. Specifically, we:

- Experimented with batch sizes and epochs: testing and training with different combinations of batch sizes (16, 32, 64) and epochs (10, 15, 20).
- Incorporated batch normalization and dropout: improving generalization and reducing overfitting.
- Balanced the datasets: augmenting or reducing the datasets to ensure balance both within individual datasets and across datasets.

## Results Storage
The results from each model on each dataset are stored as images in the branches named after the respective models. These images include:

1. **Confusion Matrices:** Visualizing performance for each dataset and batch size/epoch combination (16-10, 32-15, 64-20).
2. **Summary Tables:** Providing an overview of the model's performance metrics (accuracy, precision, recall, and F1 scores) across all datasets.

These results are referenced throughout this document to highlight key findings and comparisons.

## Datasets
The datasets used for this project were sourced from Kaggle. While some of the datasets originally included samples for numbers, only the folders containing letters were considered. Below is an overview of the datasets:

- **ASL Dataset:**
  - The largest dataset with 26 classes (A-Z).
  - Each class contains 3,000 images.

- **LIS Dataset:**
  - The smallest and most unbalanced dataset with 22 classes.
  - Class sizes range from 240 to 330 images.

- **BSL Dataset:**
  - The most distinct and noisy dataset with 23 classes.
  - Each class contains 1,000 images.
  - Some images were blurred or showed more of the person performing the sign rather than focusing on the hands.

The letters that were not represented in the dataset (H, J, Y for the BSL and G, J, S, Z for the LIS) were omitted by the creators since these signs needed movement to be performed correctly.

The BSL dataset, due to the considerable size of its files, has been modified using `Dataset_Resizing.py`: the files have been converted from `.png` to `.jpg`, with their size reduced by 75%.

Due to GitHub limitations on file size, the datasets have been uploaded on a shared Google Drive folder (https://drive.google.com/drive/folders/1t2bYdvcHNPQheHlmNWSSlvmu2JbSp4wc).

### Dataset Balancing
To address class imbalances and differences across datasets, we manipulated the data using `Dataset_Balancing.py`. This resulted in six balanced datasets:

- **ASL(1000):** Reduced to 1,000 images per class.
- **ASL(300):** Reduced to 300 images per class.
- **BSL(1000):** Original dataset with no changes.
- **BSL(300):** Reduced to 300 images per class.
- **LIS(300):** Augmented to 300 images per class.
- **LIS(1000):** Augmented to 1,000 images per class.

### Combined Dataset
To enable cross-dataset learning, the ASL(300), BSL(300), and LIS(300) datasets were merged using `create_combined_dataset.py`. This resulted in a Combined Dataset, containing balanced samples from all three sign language alphabets.

## ResNet50
ResNet50 is a deep learning model launched by Microsoft Research. Its 50-layer architecture focuses on residual learning and involves:

- Input layer: Accepts (224, 224, 3) tensors (224 × 224 RGB images).
- Convolutional layer: 7×7 filter, 64 filters, and a stride of 2, to detect features and generate feature maps.
- 3×3 max pooling layer: Reduces spatial dimensions, retaining important information while reducing computational load.
- 4 residual blocks: Each with a specific number of units of 1×1 convolution, 3×3 convolution, and 1×1 convolution with variable numbers of filters.
- Average pooling layer: Reduces the spatial dimensions of the feature maps to a single value per feature, simplifying the architecture.
- Fully connected layer with 1,000 neurons.
- Softmax activation in the output layer: Produces the model's predictions.

The key feature of this model is the use of residual blocks, which connect the activations of previous layers with the next one. This allows the network to skip intermediate layers, mitigating the vanishing/exploding gradient problem, enabling smoother training and faster convergence.

### Results
To thoroughly test the capabilities of ResNet50, which has approximately twice the number of trainable layers compared to the other architectures, the model was evaluated on datasets of both sizes: 300 images per class (as the other architectures) and 1,000 images per class.

To maximize performance, each of the 7 datasets was tested with 6 trials, varying by 3 factors:

- **Learning rate:** 1e-4 or 1e-5.
- **Base model layers unfrozen:** None or only BatchNormalization layers. BatchNormalization layers contain learned parameters (mean and variance) that align the model with the original dataset's distribution. By unfreezing these layers, the model can adapt to the new dataset, improving accuracy, especially for large datasets.
- **Dense regularization:** Adding L2 regularization to the Dense layer. Dense layer regularization, through L2 penalties, discourages large weight magnitudes, which can lead to overfitting.

Trials where BatchNormalization layers are unfrozen and Dense regularization with L2 penalty is added generally improve results, especially for large datasets, stabilizing the performance by reducing overfitting.

From this table, the following observations can be made:

- **LIS(1000):** Achieved high accuracy without the need for a lower learning rate. Its sharp contrast between the hand and the clean and uniform black background made feature extraction straightforward.
- **LIS(300):** Despite being smaller, also performed well due to the clarity and simplicity of the images.
- **ASL(1000):** Faced challenges due to lower resolution and poor lighting, leading to low contrast between the hand and background. Using BatchNormalization layers with a lower learning rate allowed the model to adapt slowly to the noise and complexity of the dataset.
- **BSL datasets:** Posed the greatest difficulty due to complex backgrounds and some gestures involving full-body poses or contextual elements. The increased complexity required slower optimization to prevent the model from focusing on irrelevant features in the background.
- **Combined dataset:** Showcased the model's ability to learn and generalize from diverse input spaces, achieving almost similar results regardless of dataset size (1,000 or 300). Achieved accuracy ranged between 89%-97%, with challenges arising in distinguishing similar gestures:
  - BSL: M/N
  - ASL: R/U/V
  - LIS: M/N, R/U

## EfficientNetV2
EfficientNetV2 is a CNN architecture based on MobileNetV2. It employs a scaling method that uniformly adjusts the network's width, depth, and resolution using a set of fixed scaling coefficients. To achieve better accuracy and efficiency, it balances these three dimensions during scaling.

### Compound Scaling Method
1. Fix φ = 1 and perform a small grid search to determine the values of α, β, and γ.
2. Fix α, β, and γ as constants, and scale up the baseline network using different values of φ.

The equations used to describe depth, width, and resolution are as follows:

- **Depth:** d = α^φ
- **Width:** w = β^φ
- **Resolution:** r = γ^φ

The architecture also incorporates squeeze-and-excitation blocks:

1. **Global Average Pooling Layer:** Reduces the spatial dimensionality of the feature maps.
2. **Fully Connected Module:** A small two-layer fully connected network to model the interdependencies between channels.
3. **Rescaling:** Adjusts the original feature maps based on the learned channel interdependencies.

### Results
- **ASL Dataset:** High performance with minimal differences between batches. The worst-performing batch still achieved results above 0.60.
- **BSL Dataset:** Slightly lower results compared to ASL but still high overall, with some confusion between letters M and N.
- **LIS Dataset:** Similar performance to ASL, with tendencies to misclassify F as B and C as O.
- **Combined Dataset:** High performance across all batches.

## MobileNetV2
MobileNetV2 is a CNN architecture based on an inverted residual structure, where residual connections occur between the bottleneck layers.

### Architecture
1. **Input:** An input image of size 224×224×3.
2. **Initial Convolution:** A convolutional layer (conv2d) with a stride of 2, producing an output with 32 channels.
3. **Bottleneck Layers:** A series of 19 bottleneck layers. Each bottleneck block is defined by:
   - Expansion block: Expands the number of channels.
   - Depthwise convolution block: Applies a 3×3 kernel to each channel independently.
   - Linear reduction block: Reduces the number of channels.
   - These blocks are repeated based on the configuration.
4. **1×1 Convolution:** A pointwise convolution.
5. **Average Pooling:** A 7×7 average pooling layer.
6. **Final Output:** A 1×1 convolution produces the final output.

### Results
- **ASL Dataset:** Acceptable results, but the model fails to recognize I and struggles with E, Y, V, and X.
- **BSL Dataset:** Slightly worse than ASL, with difficulties identifying A, O, and T.
- **LIS Dataset:** Struggles with F, M, P, Q, U, V, and X.
- **Combined Dataset:** Results were still similar, but the model fails to recognize several letters.

## VGG16 & VGG19
VGG is a family of deep convolutional neural networks known for its simple architecture. These models vary by depth and number of layers with tunable parameters, offering four variants (11, 13, 16, and 19 layers). Despite differences, they share a common structure:

- **Input layer:** Accepts a colored RGB image (224×224×3 tensor).
- **Convolutional layers:** Filters with a small receptive field (3×3) and stride of 1, with row and column padding.
- **Max pooling layers:** Use a 2×2 kernel with a stride of 2 to reduce spatial dimensions.
- **Fully connected layers:** 3 layers to capture complex patterns.
- **Output layer:** A softmax activation function for classification.

### Results
1. **LIS(300):** The simpler architecture of VGG16 proved sufficient, with VGG19 showing no significant improvement.
2. **BSL(300):** Best results were achieved for the 32/15 configuration, with VGG19 resolving common misclassifications (e.g., letter "C").
3. **ASL(300):** Training with fewer epochs led to underfitting for VGG19, whereas optimal results were observed with the 32/15 configuration.
4. **Combined Dataset:** VGG16 unexpectedly outperformed VGG19.

## Final Results
### Summary
- **ResNet50:** Emerged as the top-performing model for both individual and combined datasets.
- **VGG19:** Consistently achieved second-best performance for individual datasets.
- **EfficientNetV2:** Demonstrated strong performance for the combined dataset.

### Dataset-Specific Observations
- **LIS(300):** VGG19 achieved the best performance (accuracy = 0.98), followed by ResNet50 (accuracy = 0.96).
- **BSL(300):** ResNet50 led with an accuracy of 0.97, while VGG19 performed similarly (accuracy = 0.96).
- **ASL(300):** ResNet50 outperformed others with an accuracy of 0.96, followed by VGG19.
- **Combined Dataset:** ResNet50 maintained its robustness, while others showed lower performance (e.g., MobileNetV2 with accuracy < 0.8).

### Recommendations
- **ResNet50:** For tasks requiring high accuracy and robustness across diverse datasets.
- **VGG19:** For datasets where deeper feature learning is advantageous.
- **EfficientNetV2:** A balanced choice for mixed datasets.

## Works Cited
1. Grassknoted. ASL Alphabet. Kaggle. [ASL Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. Nisopoli, Nicholas. LIS Dataset. Kaggle. [LIS Dataset](https://www.kaggle.com/datasets/nicholasnisopoli/lisdataset)
3. Tatepe, Eren. BSL Numbers and Alphabet Hand Position for MediaPipe. Kaggle. [BSL Dataset](https://www.kaggle.com/datasets/erentatepe/bsl-numbers-and-alphabet-hand-position-for-mediapipe)
4. K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.
5. He, K., Zhang, X., Ren, S., and Sun, J. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. arXiv, 2015.
6. Thakur, Ayush. Keras Dense Layer: How to Use It Correctly. Weights & Biases.
7. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.C. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." arXiv preprint arXiv:1801.04381 (2019).
8. Tan, M., Le, Q.V. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." arXiv preprint arXiv:1905.11946 (2019).
9. Simonyan, K., "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv preprint arXiv:1409.1556 (2014).
