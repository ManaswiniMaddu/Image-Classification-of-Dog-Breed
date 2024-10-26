# Image-Classification-of-Dog-Breed
Conducted exploratory data analysis and applied image classification techniques, including Convolutional Neural Networks(CNN), to predict dog breeds from images, achieving a highest accuracy of 94.11% through data rescaling, augmentation, and CNN optimization.

# Project Overview
The purpose of this project is to classify dog breeds from images using Convolutional Neural Networks(CNNs). By comparing a basic CNN with a deeper, more complex CNN, we aim to determine which model architecture performs best on a subset of dog breeds. This project was executed using a Kaggle Notebook environment, utilizing Kaggle's dataset for both training and testing.

# Hypothesis
The hypothesis is that the basic CNN model will outperform the deeper model due to the limited dataset (only two breeds). The primary metric for performance evaluation is accuracy, along with other metrics such as precision, recall, and F1-score.

# Dataset
1. Source: Kaggle dog breed dataset, containing images across 120 breeds.
2. Filtered Dataset: We focused on two breeds, Boston Bull and Dingo, totaling 167 images.
3. Labels: A CSV file with image labels is used for supervised training.

# 5V Model Analysis
1. Volume: Total of 10222 images across all breeds; subset includes 167 images for two breeds.
2. Velocity: This dataset can scale up by including more images, increasing data processing speed.
3. Variety: 120 dog breeds originally; subset limited to Boston Bull and Dingo.
4. Veracity: The images have good quality, with minor noise handled in preprocessing.
5. Value: Adds value by automating breed identification from images.

# Working
1. Data Preparation
   - Splitting: Data is split into training (90%) and testing (10%) sets, yielding 150 training images and 17 test images.
   - Label Mapping: Each breed is assigned a binary label (Boston Bull = 0, Dingo = 1).
   - Image Resizing and Rescaling*: Images are resized for uniformity and scaled to values between [0, 1] for improved model performance.
   - Data Augmentation: Techniques like rotation and flipping are applied to increase training data diversity.
2. Model Architectures
   - Basic CNN: A standard CNN with essential layers (convolutional, pooling, and fully connected layers) to extract features and classify images.
   - CNN with More Hidden Layers: A deeper CNN to capture complex features, but it risks overfitting on the limited dataset.
3. Training the Models
   - Each model is trained on the preprocessed images, using training data to update weights and biases in the neural network.
   - Loss Function: Binary Cross-Entropy loss is used to measure classification error.
   - Optimizer: The Adam optimizer adjusts model weights to minimize the loss function.
4. Evaluation and Performance Metrics
   - Testing: After training, each model is evaluated on the test dataset to gauge performance on unseen images.
   - Metrics: Accuracy, precision, recall, and F1-score are calculated to evaluate model effectiveness. Accuracy is the main metric for comparing models.

# Exploratory Data Analysis (EDA)
Key analysis steps include:
- Checking for label consistency with images.
- Removing any null values to improve accuracy.
- Ensuring all images have the same dimensions and are RGB.
- Analyzing per-class image distribution for training and testing data balance.

# Results and Conclusion
- Basic CNN Model: Achieved an accuracy of 94.11% on test data, making it the better performer with this subset.
- Deeper CNN Model: Achieved an accuracy of 70.5%, possibly due to overfitting on a small dataset.
- Conclusion: For a small, limited dataset, simpler CNN architectures are more effective. Expanding the dataset to more breeds and samples would likely improve deeper models performance.

# Getting Started
1. Clone this repository.
2. Install necessary dependencies (tensorflow, keras, numpy, matplotlib, etc.).
3. Download the dataset from Kaggle and place it in the "data" directory.
4. Run the Jupyter Notebook to train and evaluate the models.
