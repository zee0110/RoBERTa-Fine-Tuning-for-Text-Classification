# RoBERTa-Fine-Tuning-for-Text-Classification

### README

#### RoBERTa Fine-Tuning for Text Classification

This code demonstrates the fine-tuning of a RoBERTa (Robustly optimized BERT approach) model for text classification using PyTorch. The model is trained on a dataset containing text reviews and corresponding ratings, aiming to predict the ratings based on the review text.

#### Prerequisites

Ensure you have Python installed on your system along with the necessary libraries:
- torch
- transformers
- pandas
- tqdm
- scikit-learn

You can install the required libraries using pip:

```
pip install torch transformers pandas tqdm scikit-learn
```

#### Usage

1. Clone the repository or download the code files.
2. Ensure you have the dataset file "fullData.csv" in the specified directory.
3. Run the code using a Python interpreter.
4. The code reads the dataset and preprocesses it, dropping rows with missing values.
5. It tokenizes the text data using the RoBERTa tokenizer and prepares the dataset for training and validation.
6. The RoBERTa model for sequence classification is initialized and compiled with an optimizer and loss function.
7. The model is trained on the training dataset with a specified number of epochs.
8. After training, the model is evaluated on the validation dataset to assess its accuracy.
9. The model is further evaluated on a separate test dataset to calculate accuracy, precision, recall, and F1 score.
10. The trained model is saved to a specified directory.

#### Features

- Loads and preprocesses text classification dataset.
- Tokenizes text data using the RoBERTa tokenizer.
- Fine-tunes a pre-trained RoBERTa model for text classification.
- Compiles the model with an optimizer and loss function.
- Trains the model on the training dataset and evaluates its performance on the validation dataset.
- Evaluates the model's accuracy, precision, recall, and F1 score on the test dataset.
- Saves the trained model for future use.

