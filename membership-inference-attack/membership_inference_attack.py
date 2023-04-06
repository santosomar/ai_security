import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocessing the dataset and creating a simple classifier model:
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

# Training the classifier model on the CIFAR-10 dataset:
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Creating a membership oracle and attack model:

# Define the membership oracle
def membership_oracle(data, labels):
    predictions = model.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)
    return (predicted_labels == labels.squeeze()).astype(int)

# Prepare the training data for the attack model
shadow_train_data = x_train[:1000]
shadow_train_labels = y_train[:1000]
shadow_test_data = x_test[:1000]
shadow_test_labels = y_test[:1000]

train_membership_labels = membership_oracle(shadow_train_data, shadow_train_labels)
test_membership_labels = membership_oracle(shadow_test_data, shadow_test_labels)

attack_train_data = np.concatenate([shadow_train_data, shadow_test_data])
attack_train_labels = np.concatenate([train_membership_labels, test_membership_labels])

# Create a simple attack model using logistic regression
scaler = StandardScaler()
attack_train_data_flat = scaler.fit_transform(attack_train_data.reshape(len(attack_train_data), -1))
attack_model = LogisticRegression()
attack_model.fit(attack_train_data_flat, attack_train_labels)

# Evaluating the performance of the attack model to determine the membership status of data points:
# Prepare the evaluation data for the attack model
eval_data = x_test[1000:2000]
eval_labels = y_test[1000:2000]

ground_truth_membership = np.zeros(len(eval_data))
predicted_membership = membership_oracle(eval_data, eval_labels)

eval_data_flat = scaler.transform(eval_data.reshape(len(eval_data), -1))
membership_proba = attack_model.predict_proba(eval_data_flat)[:, 1]

accuracy = accuracy_score(ground_truth_membership, predicted_membership)
print("Membership inference attack accuracy: {:.2f}".format(accuracy))

