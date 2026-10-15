# deberta--emotion-classifier
# Multi-Label Emotion Classification with DeBERTa-v3

## Overview

This project builds a multi-label text classification pipeline to predict five emotions from text: **Anger, Joy, Fear, Sadness, and Surprise**.

Since a text can express multiple emotions simultaneously, this problem is treated as **multi-label classification** rather than single-label classification.

## Project Workflow

1. Load the dataset and create a training/validation split.
2. Tokenize text using the pretrained DeBERTa-v3 tokenizer.
3. Prepare PyTorch datasets, batches, and data loaders.
4. Fine-tune a DeBERTa model with a five-label classification head.
5. Evaluate validation performance using Macro F1.
6. Tune a separate prediction threshold for each emotion.
7. Generate predictions on the competition test set and create a submission CSV.

## Model and Training Configuration

| Parameter               | Value                       |
| ----------------------- | --------------------------- |
| Pretrained model        | `microsoft/deberta-v3-base` |
| Maximum sequence length | 256 tokens                  |
| Batch size              | 16                          |
| Optimizer               | AdamW                       |
| Learning rate           | `1e-5`                      |
| Loss function           | `BCEWithLogitsLoss`         |
| Evaluation metric       | Macro F1                    |

## Live Demo

[Click here to try the deployed application](https://huggingface.co/spaces/Deepak8409/deberta-emotion-classifier)