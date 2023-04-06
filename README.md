# AI Security Proof-of-Concept Attacks

This repository contains multiple examples of proof-of-concept (PoC) attacks against AI systems, demonstrating various security vulnerabilities in machine learning models. These examples aim to provide a better understanding of potential risks and help researchers and developers build more secure AI systems.

## Table of Contents

1. [Model Stealing Attack on Sentiment Analysis Model](#model-stealing-attack)
2. [Adversarial Attack on Image Classifier](#adversarial-attack)
3. [Membership Inference Attack on ML Models](#membership-inference-attack)
4. [Model Inversion Attack on ML Models](#model-inversion-attack)
5. [Data Poisoning Attack on ML Models](#data-poisoning-attack)

## Requirements

- Python 3.6 or higher
- TensorFlow 2.x
- PyTorch
- NumPy
- scikit-learn
- foolbox
- matplotlib

## Installation

To install the required libraries, run the following command:

```
pip install tensorflow numpy scikit-learn foolbox matplotlib torch torchvision
```

Usage
Clone this repository to your local machine.
Install the required libraries as mentioned above.
Navigate to the desired example folder and follow the instructions in the corresponding README.md file to run the attack.
<a name="model-stealing-attack"></a>

1. Model Stealing Attack on Sentiment Analysis Model
This example demonstrates a model stealing attack on a simple sentiment analysis model using the IMDb movie review dataset. The attacker queries the target model and trains a replica based on its responses.



<a name="adversarial-attack"></a>

2. Adversarial Attack on Image Classifier
This example shows an adversarial attack on an image classifier using the Fast Gradient Sign Method (FGSM) to generate adversarial examples that can fool the model.


<a name="membership-inference-attack"></a>

3. Membership Inference Attack on ML Models
This example demonstrates a membership inference attack on machine learning models, in which an attacker tries to determine if a given data point was part of the model's training dataset.


<a name="model-inversion-attack"></a>

4. Model Inversion Attack on ML Models
This example illustrates a model inversion attack, where an attacker attempts to reconstruct input data from a machine learning model's predictions or internal representations.


<a name="data-poisoning-attack"></a>

5. Data Poisoning Attack on ML Models
This example demonstrates a data poisoning attack on machine learning models, in which an attacker injects malicious data into the training dataset to compromise the model's performance or introduce specific vulnerabilities.



