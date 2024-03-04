def show_char(index):
    # Display the image with smaller window size and in grayscale
    plt.figure(figsize=(4, 4))  # Set the figure size to a smaller window size
    plt.imshow(X_train_orig[index], cmap='gray')  # Display the image in grayscale
    plt.axis('off')  # Turn off axis labels
    plt.show()