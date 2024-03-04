from data_preprocess import create_train_test_data, \
    split_train_test_data, preprocess_data
from model import HahuNet


# create the training and test datasets from the images in the directory
train_dataset, test_dataset = create_train_test_data('../dataset')
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = \
    split_train_test_data(train_dataset, test_dataset)


X_train, Y_train, X_test, Y_test = preprocess_data(
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)


# Define the model parameters
input_shape = X_train.shape[1:]
num_classes = Y_train.shape[1]
epochs = 20
batch_size = 64


# Create the model instance
model = HahuNet(input_shape, num_classes)
print("Model initialized")
# Train the model
model.train(X_train, Y_train, epochs, batch_size)
print("Model trained")
# Evaluating the model
loss, accuracy = model.evaluate(X_test, Y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
