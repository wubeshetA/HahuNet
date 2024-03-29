# HahuNet

HahuNet is a Convolutional Neural Network (CNN) to recognize handwritten Amaric characters. The model is trained on a dataset of 28x28 grayscale images with over 32,000 data points.

HahuNet is a project that I started to learn about deep learning and computer vision. The name "Hahu-Net" is found by combining Ha and hu, which are the sounds of the first and second characters in the amharic alphabet and Net is a common suffix for neural networks. In Ethiopia Hahu (ሀሁ) is a common traditional name to refer to the entire Amharic alphabet.

 - Here is a picture of the Amharic alphabet:

 ![Amharic Alphabet](https://raw.githubusercontent.com/wubeshetA/HahuNet/main/alphabets2.jpg)

## Model Architecture

The model architecture is defined in the `HahuNet2` class. It consists of the following layers:

- Convolutional layer with 16 filters of size 3x3, ReLU activation
- Max pooling layer with pool size 2x2
- Convolutional layer with 32 filters of size 7x7, ReLU activation
- Max pooling layer with pool size 2x2
- Convolutional layer with 64 filters of size 9x9, ReLU activation
- Max pooling layer with pool size 2x2
- Flatten layer
- Dense layer with 512 units, ReLU activation
- Dropout layer with rate 0.5
- Output Dense layer with softmax activation

The model uses categorical cross-entropy as the loss function and accuracy as the metric.

## Installation
```bash
> git clone https://github.com/wubeshetA/HahuNet.git
> cd HahuNet
> pip install -r requirements.txt
```

## Evaluation

Currently the Model achieved 86.9% accuracy on the validation and test set. It also achieved 97% accuracy on the training set.
I look forward to improving the model's performance on the validation and test sets.

## Future Work

- Future work includes:
  - Improving the model's performance on the validation and test sets.
  - Build different digital tools that can be purposeful in the case of of identifying Amaric letters. This might include but not limited
    to a apps that that can scan a large paper of handwritten Amaric scripts and convert them to digital text.

## Acknowledgements

- Thank you [Fetulhak Abdurahman](https://github.com/Fetulhak/) for collecting and organizing the handwritten Amharic characters dataset.

## Author

[Wubeshet Yimam](https://linkedin.com/in/wubeshet/).
