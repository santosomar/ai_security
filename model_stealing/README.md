# Model Stealing Attack on Sentiment Analysis Model

This repository contains a Python script demonstrating a model stealing attack on a simple sentiment analysis model using the IMDb movie review dataset. The attacker creates a replica of the target model by querying it repeatedly and training a new model based on the target model's responses.

## Requirements

- Python 3.6 or higher
- TensorFlow 2.x
- NumPy

To install the required libraries, run the following command:

```
pip install tensorflow numpy
```

## Dataset
The IMDb movie review dataset is used in this example. It is available within the TensorFlow library and can be loaded using the following code:

```
from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
```

## Proof-of-Concept (POC) Script
The script is organized into six sections:
1. Importing required libraries and loading the IMDb dataset.
2. Splitting the dataset into training data for the target model, training data for the attacker's model, and test data.
3. Creating and training the target model.
4. Querying the target model to obtain labels for the attacker's model.
5. Creating and training the attacker's model.
6. Evaluating the performance of the attacker's model.

