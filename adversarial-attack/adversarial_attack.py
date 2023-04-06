import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from foolbox import TensorFlowModel, accuracy, samples
from foolbox.attacks import LinfFastGradientAttack

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocessing the dataset and creating the image classifier model:

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10)
])

#Training the image classifier model on the CIFAR-10 dataset:

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# Generating adversarial examples using the Fast Gradient Sign Method (FGSM):
# Prepare the model for foolbox
fmodel = TensorFlowModel(model, bounds=(0, 1))

# Get a batch of test images and labels
images, labels = samples(fmodel, dataset="cifar10", batchsize=20)

# Apply the FGSM attack
attack = LinfFastGradientAttack()
epsilons = [0.1]
adversarials, _, success = attack(fmodel, images, labels, epsilons=epsilons)


# Evaluating the performance of the image classifier on the original test dataset and adversarial examples:

# Evaluate the original accuracy
original_accuracy = accuracy(fmodel, images, labels)

# Evaluate the adversarial accuracy
adversarial_accuracy = accuracy(fmodel, adversarials, labels)

print("Original accuracy:", original_accuracy)
print("Adversarial accuracy:", adversarial_accuracy)

# Visualize the original images, adversarial images, and the differences
plt.figure(figsize=(15, 5))

for i in range(10):
    plt.subplot(3, 10, i + 1)
    plt.imshow(images[i])
    plt.axis("off")

    plt.subplot(3, 10, 10 + i + 1)
    plt.imshow(adversarials[i])
    plt.axis("off")

    plt.subplot(3, 10, 20 + i + 1)
    plt.imshow(np.abs(adversarials[i] - images[i]) * 255)
    plt.axis("off")

plt.show()





