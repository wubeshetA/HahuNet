import matplotlib.pyplot as plt
import pickle


def show_char(index, X_train_orig):
    """Display the image at the given index in the training dataset

    Args:
        index (int): The index of the image in the training dataset
        X_train_orig (numpy.ndarray): The training dataset
    """
    # Display the image with smaller window size and in grayscale
    plt.figure(figsize=(4, 4))  # Set the figure size to a smaller window size
    # Display the image in grayscale
    plt.imshow(X_train_orig[index], cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.show()


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def save_model(model, model_name):
    """Save the model to a file

    Args:
        model (tf.keras.Model): The model to save
        model_name (str): The name of the file to save the model
    """
    # Save the trained model to a pickle file
    with open(model_name, 'wb') as file:
        pickle.dump(model, file)


def predict_and_visualize(model, X_test, Y_test, num_samples=None):
    # Reshape the input data to match the model's input shape
    X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

    if num_samples is not None:
        # Select the first num_samples samples for prediction and visualization
        X_test_reshaped = X_test_reshaped[:num_samples]
        Y_test = Y_test[:num_samples]

    predictions = model.model.predict(X_test_reshaped)

    # Get the predicted class for each sample
    predicted_classes = predictions.argmax(axis=1)

    # Get the actual class for each sample
    actual_classes = Y_test.argmax(axis=1)

    # Visualize predicted and actual labels with images
    fig, axes = plt.subplots(5, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(len(predicted_classes)):
        ax = axes[i]
        ax.imshow(X_test_reshaped[i].reshape(28, 28), cmap='gray')
        ax.set_title(
            f"Predicted: {predicted_classes[i]}\nActual: {actual_classes[i]}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
