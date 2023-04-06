# Adversarial Attack on Image Classifier
This section contains an example of an adversarial attack on an image classifier using the Fast Gradient Sign Method (FGSM) to generate adversarial examples that can fool the model. The example demonstrates how even slight perturbations to the input data can lead to incorrect predictions.

## Requirements
- Python 3.6 or higher
- TensorFlow 2.x
- NumPy
- matplotlib
- foolbox

To install the required libraries, run the following command:
```
pip install tensorflow numpy matplotlib foolbox
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
- Run the `adversarial_attack.py` script to simulate the adversarial attack. This script will train the image classifier on the CIFAR-10 dataset, generate adversarial examples using FGSM, and evaluate the performance of the classifier on the original test dataset and adversarial examples.

The script is organized into five sections:
1. Importing required libraries and loading the CIFAR-10 dataset.
2. Preprocessing the dataset and creating the image classifier model.
3. Training the image classifier model on the CIFAR-10 dataset.
4. Generating adversarial examples using the Fast Gradient Sign Method (FGSM).
5. Evaluating the performance of the image classifier on the original test dataset and adversarial examples.
