## Context
The GitHub repository for this project is available here: Hand Wash Step Classifier Repository

This dataset is a curated subset of the original Hand Wash Dataset on Kaggle, designed for fine-grained action recognition—particularly in healthcare settings like hospitals and schools.

While common action recognition datasets (e.g., UCF101, KTH) focus on large, distinct movements, handwashing procedures based on the WHO 7-step guide involve subtle, repetitive hand motions that are more challenging to detect.

To better simulate real-world conditions, the data features:

Fixed camera positions

Static backgrounds

Varying lighting

Different individuals and settings

## Dataset
The original dataset contains 292 videos and 3,504 clips spanning 12 handwashing actions. For practical use and improved robustness, it was simplified to 6 classes by:

Merging left/right hand variants

Removing Step 7 due to inconsistency

This 6-class version is ideal for real-time feedback systems where simplicity and speed are essential.

## Model
We use a ResNet-18 model pretrained on ImageNet, then fine-tuned on the modified dataset. Despite environmental variations, the model performs well in classifying the six handwashing steps.

## Download
Download the modified 6-class Hand Wash Dataset here: [Download from Google Drive](https://drive.google.com/file/d/1nz0l8r07tl1-mdDjgE7Qcx3RUWJ8Sed2/view?usp=sharing)

Original Kaggle source:
https://www.kaggle.com/datasets/realtimear/hand-wash-dataset
