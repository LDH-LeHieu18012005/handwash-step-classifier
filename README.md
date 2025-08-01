Context
The GitHub repository that contains the application of this dataset can be found here:
[Hand Wash Step Classifier Repository](https://github.com/LDH-LeHieu18012005/handwash-step-classifier)

This dataset is a sample extracted from the complete Hand Wash Dataset originally published on Kaggle. It is intended to serve as a benchmark for fine-grained action recognition tasks, particularly in healthcare settings.

Most existing public action recognition datasets—such as UCF101, KTH, and UCF Sports—are designed for generic or large-motion actions like sports or daily routines. These datasets involve significant visual differences between classes, often with distinct environments and large movements, making them less suitable for subtle, repetitive actions.

In contrast, handwashing procedures, especially those based on the WHO's 7-step guideline, involve minimal movements and small hand position changes between steps. These finer-grained actions pose challenges for traditional action recognition models.

Moreover, real-world applications typically involve:

Fixed camera positions

Static backgrounds

Varied lighting conditions

The need to simulate such realistic settings justifies the creation and adaptation of this dataset.

Content
The original Hand Wash Dataset consists of 292 videos, divided into 3,504 action clips, spanning 12 specific actions derived from the WHO’s 7-step handwashing process. These clips were captured across diverse environments to ensure model generalization, with variations in:

Illumination

Backgrounds

Camera angles and positions

Field of view

Individuals performing the actions

This dataset was designed to simulate real-world conditions such as static camera setup, real-time feedback, and changing environments, with the goal of recognizing subtle hand movements during the washing procedure.

The 12 original classes are:

Step 1

Step 2 Left

Step 2 Right

Step 3

Step 4 Left

Step 4 Right

Step 5 Left

Step 5 Right

Step 6 Left

Step 6 Right

Step 7 Left

Step 7 Right

✂Dataset Adaptation
To streamline the classification task and improve robustness, the dataset was modified from 12 classes down to 6 main steps, by:

Merging left and right actions for each step.

Removing Step 7 entirely, which is often inconsistently performed and harder to annotate.

This adjusted 6-class dataset better reflects practical use-cases where simplicity and speed are preferred, such as in real-time feedback systems in hospitals or schools.

Model
The classification model is based on ResNet-18, a deep convolutional neural network pretrained on ImageNet and fine-tuned using the modified 6-class Hand Wash Dataset. The model achieves high accuracy in distinguishing between the six handwashing steps, even in different backgrounds and lighting conditions.

Download the Modified Dataset
You can download the modified 6-class Hand Wash Dataset (prepared by me based on the original Kaggle dataset) from Google Drive here:
[🔗 Download Hand Wash Dataset (6 Classes)](https://drive.google.com/file/d/1nz0l8r07tl1-mdDjgE7Qcx3RUWJ8Sed2/view?usp=sharing)

Original Kaggle source:
🔗 https://www.kaggle.com/datasets/realtimear/hand-wash-dataset
