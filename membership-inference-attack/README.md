# Membership Inference Attack on ML Models
This repository contains an example of a Membership Inference Attack on machine learning models. In this attack, an adversary aims to determine whether a given data point was part of the model's training dataset. This example demonstrates potential privacy risks associated with machine learning models and highlights the importance of considering privacy during the development process.

## Requirements
- Python 3.6 or higher
- TensorFlow 2.x
- NumPy
- scikit-learn

To install the required libraries, run the following command:
```
pip install tensorflow numpy scikit-learn
```

## Dataset
The CIFAR-10 dataset is used in this example. It is available within the TensorFlow library and can be loaded using the following code:

```
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```
## Usage
- Clone this repository to your local machine.
- Install the required libraries as mentioned above.
- Run the membership_inference_attack.py script to simulate the membership inference attack:

This script will train a simple classifier on the CIFAR-10 dataset, create a membership oracle and attack model, and evaluate the performance of the attack model to determine the membership status of the data points.

The script is organized into five sections:

1. Importing required libraries and loading the CIFAR-10 dataset.
2. Preprocessing the dataset and creating a simple classifier model.
3. Training the classifier model on the CIFAR-10 dataset.
4. Creating a membership oracle and attack model.
5. Evaluating the performance of the attack model to determine the membership status of data points.