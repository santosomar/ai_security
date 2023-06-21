```
'''
Example of a model stealing stack 
Part of the Beyond the Algorithm book by Omar Santos and Dr. Petar Radanliev
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a target model with RandomForest
target_model = RandomForestClassifier(n_estimators=50)
target_model.fit(X_train, y_train)

# Create surrogate model architecture
class SurrogateModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SurrogateModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Set hyperparameters
input_size = 4
hidden_size = 50
num_classes = 3
num_epochs = 100
learning_rate = 0.01

# Instantiate the surrogate model
surrogate_model = SurrogateModel(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=learning_rate)  

# Train the surrogate model using the target model's predictions
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(X_train).float()
    labels = torch.from_numpy(target_model.predict(X_train))
    
    # Forward pass
    outputs = surrogate_model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        
# Test the surrogate model
inputs = torch.from_numpy(X_test).float()
labels = torch.from_numpy(y_test)
outputs = surrogate_model(inputs)
_, predicted = torch.max(outputs.data, 1)
accuracy = (labels == predicted).sum().item() / len(y_test)
print('Accuracy of the surrogate model on the test data: {} %'.format(accuracy*100))
```
# Explaination 
This script shows how you can train a neural network (surrogate model) to mimic another model (target model, a RandomForest in this case).

This is how it works:

- First, it loads the iris dataset using sklearn's `load_iris()` function, which returns a dictionary-like object containing the features (`data`) and labels (`target`).
- It then splits the dataset into a training set and a test set using `train_test_split()`. The test set is 20% of the total dataset, as specified by `test_size=0.2`.
- A RandomForest model (the target model) is then created and trained using the training data. The RandomForest is an ensemble learning method for classification (or regression) that operates by constructing multiple decision trees during training.

The next part of the code is dedicated to creating and training a surrogate model, which is a simple feed-forward neural network:

- The `SurrogateModel` class defines the architecture of the model. It has one hidden layer (`fc1`), an activation function (`ReLU`), and an output layer (`fc2`).
- After setting some hyperparameters (input size, hidden size, number of classes, number of epochs, and learning rate), an instance of `SurrogateModel` is created.
- The loss function is set to cross-entropy loss, which is suitable for multi-class classification tasks. The optimizer is set to Adam, a popular choice for training neural networks.

The surrogate model is then trained with a unique twist:

- Instead of training the surrogate model using the original labels, it is trained using the predictions of the target model (RandomForest). The idea is to make the surrogate model learn to mimic the behavior of the target model.
- The inputs are converted to PyTorch tensors, and then the forward pass is done by passing the inputs to the surrogate model.
- The loss is calculated by comparing the output of the surrogate model with the predictions of the target model.
- Then, gradients are calculated by calling `loss.backward()`, and the optimizer updates the weights of the surrogate model.

Finally, the surrogate model is evaluated on the test data:

- The test data is converted to a tensor and fed to the surrogate model.
- The maximum of the output (since it's a multi-class classification task) is taken as the predicted label, and this is compared with the actual labels to compute the accuracy of the surrogate model.

