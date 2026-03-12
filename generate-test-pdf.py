"""
Generate a test PDF with ML content for RAG testing.
Run: python generate_test_pdf.py
Output: data/raw_pdfs/ml_fundamentals.pdf
"""

import fitz  # PyMuPDF
import os

os.makedirs("data/raw_pdfs", exist_ok=True)

doc = fitz.open()

pages = [
    {
        "title": "Chapter 1: Introduction to Machine Learning",
        "content": """Machine learning is a branch of artificial intelligence that focuses on building systems 
that learn from data. Instead of being explicitly programmed to perform a task, these systems identify 
patterns in data and make decisions with minimal human intervention.

There are three main types of machine learning:

1. Supervised Learning: The algorithm learns from labeled training data. Each training example has an 
input and a desired output. Common tasks include classification (predicting categories) and regression 
(predicting continuous values). Examples include spam detection, image recognition, and house price prediction.

2. Unsupervised Learning: The algorithm works with unlabeled data and tries to find hidden patterns or 
structures. Common techniques include clustering (grouping similar data points), dimensionality reduction, 
and anomaly detection. K-means clustering and Principal Component Analysis (PCA) are popular algorithms.

3. Reinforcement Learning: An agent learns by interacting with an environment and receiving rewards or 
penalties. The agent aims to maximize cumulative reward over time. Applications include game playing 
(like AlphaGo), robotics, and autonomous driving."""
    },
    {
        "title": "Chapter 2: Neural Networks and Deep Learning",
        "content": """A neural network is a computational model inspired by the structure of biological neurons. 
It consists of layers of interconnected nodes (neurons), where each connection has an associated weight.

The basic architecture includes:
- Input Layer: Receives the raw data features
- Hidden Layers: Perform computations and learn abstract representations
- Output Layer: Produces the final prediction or classification

Deep learning refers to neural networks with multiple hidden layers (typically more than two). These deep 
networks can learn hierarchical representations of data, where each layer captures increasingly abstract features.

Activation Functions are crucial components that introduce non-linearity:
- ReLU (Rectified Linear Unit): f(x) = max(0, x). Most widely used due to computational efficiency.
- Sigmoid: f(x) = 1/(1+e^-x). Outputs values between 0 and 1, useful for binary classification.
- Tanh: f(x) = (e^x - e^-x)/(e^x + e^-x). Outputs between -1 and 1, zero-centered.
- Softmax: Used in output layers for multi-class classification, converts outputs to probabilities.

Backpropagation is the algorithm used to train neural networks. It calculates the gradient of the loss 
function with respect to each weight by applying the chain rule of calculus, propagating the error 
backward through the network. The weights are then updated using an optimization algorithm like gradient descent."""
    },
    {
        "title": "Chapter 3: Training and Optimization",
        "content": """Gradient Descent is the primary optimization algorithm used to minimize the loss function 
during training. It iteratively adjusts model parameters in the direction that reduces the error.

Variants of Gradient Descent:
- Batch Gradient Descent: Computes the gradient using the entire training dataset. Stable but slow for large datasets.
- Stochastic Gradient Descent (SGD): Updates parameters using one training example at a time. Faster but noisy.
- Mini-batch Gradient Descent: Uses a small batch of examples (typically 32-256). Best of both worlds.

The Learning Rate is a critical hyperparameter that controls the step size during optimization. 
Too high: the model may overshoot the minimum and diverge. 
Too low: training becomes extremely slow and may get stuck in local minima.

Advanced Optimizers:
- Adam (Adaptive Moment Estimation): Combines momentum and adaptive learning rates. Most popular choice.
- RMSprop: Adapts learning rate for each parameter based on recent gradient magnitudes.
- AdaGrad: Adapts learning rate based on historical gradients. Good for sparse data.

Regularization Techniques prevent overfitting:
- L1 Regularization (Lasso): Adds absolute value of weights to loss. Can produce sparse models.
- L2 Regularization (Ridge): Adds squared weights to loss. Prevents any single weight from being too large.
- Dropout: Randomly deactivates neurons during training with a given probability (typically 0.2-0.5).
- Early Stopping: Monitors validation loss and stops training when it begins to increase.
- Data Augmentation: Artificially increases training data by applying transformations (rotation, flipping, etc.)."""
    },
    {
        "title": "Chapter 4: Model Evaluation",
        "content": """Proper evaluation is essential to ensure a model generalizes well to unseen data.

Train-Test Split: The dataset is divided into training data (typically 70-80%) and test data (20-30%). 
The model is trained on the training set and evaluated on the test set.

Cross-Validation: K-fold cross-validation divides data into K equal parts. The model is trained K times, 
each time using K-1 folds for training and 1 fold for validation. This provides a more robust estimate 
of model performance than a single train-test split.

Classification Metrics:
- Accuracy: Proportion of correct predictions. Can be misleading with imbalanced classes.
- Precision: Of all positive predictions, how many were actually positive? TP/(TP+FP)
- Recall (Sensitivity): Of all actual positives, how many were correctly identified? TP/(TP+FN)
- F1 Score: Harmonic mean of precision and recall. 2*(Precision*Recall)/(Precision+Recall)
- ROC-AUC: Area Under the Receiver Operating Characteristic curve. Measures discrimination ability.

Regression Metrics:
- Mean Squared Error (MSE): Average of squared differences between predicted and actual values.
- Root Mean Squared Error (RMSE): Square root of MSE. Same unit as target variable.
- Mean Absolute Error (MAE): Average of absolute differences. Less sensitive to outliers than MSE.
- R-squared: Proportion of variance in the target explained by the model. Ranges from 0 to 1.

The Bias-Variance Tradeoff:
- High Bias (Underfitting): Model is too simple, misses important patterns. High error on both training and test data.
- High Variance (Overfitting): Model is too complex, memorizes noise. Low training error but high test error.
- The goal is to find the sweet spot that minimizes total error."""
    },
    {
        "title": "Chapter 5: Popular Algorithms",
        "content": """Decision Trees split data based on feature values to make predictions. They are interpretable 
and handle both numerical and categorical data. However, they tend to overfit. The splitting criteria 
include Gini impurity and information gain (entropy).

Random Forests are ensembles of decision trees. Each tree is trained on a random subset of data and features. 
The final prediction is the majority vote (classification) or average (regression) of all trees. 
Random forests reduce overfitting compared to individual decision trees and handle high-dimensional data well.

Support Vector Machines (SVM) find the optimal hyperplane that maximizes the margin between classes. 
The kernel trick allows SVMs to handle non-linearly separable data by mapping it to a higher-dimensional space. 
Common kernels include linear, polynomial, and RBF (Radial Basis Function).

K-Nearest Neighbors (KNN) classifies a data point based on the majority class of its K nearest neighbors. 
It is a lazy learning algorithm (no training phase). Distance metrics include Euclidean, Manhattan, 
and Minkowski distance. Choosing the right K is important: too small leads to noise sensitivity, 
too large leads to over-smoothing.

Naive Bayes classifiers apply Bayes' theorem with the assumption of feature independence. 
Despite this simplifying assumption, they work surprisingly well for text classification, 
spam filtering, and sentiment analysis. Variants include Gaussian, Multinomial, and Bernoulli Naive Bayes."""
    },
]

for page_data in pages:
    page = doc.new_page(width=612, height=792)  # US Letter size

    # Title
    title_rect = fitz.Rect(50, 40, 562, 80)
    page.insert_textbox(
        title_rect,
        page_data["title"],
        fontsize=16,
        fontname="helv",
        color=(0.1, 0.1, 0.5),
    )

    # Separator line
    page.draw_line(fitz.Point(50, 85), fitz.Point(562, 85), color=(0.3, 0.3, 0.3), width=0.5)

    # Body content
    content_rect = fitz.Rect(50, 95, 562, 750)
    page.insert_textbox(
        content_rect,
        page_data["content"],
        fontsize=10,
        fontname="helv",
        color=(0, 0, 0),
    )

output_path = "data/raw_pdfs/ml_fundamentals.pdf"
doc.save(output_path)
doc.close()

print(f"PDF generated: {output_path}")
print(f"Pages: {len(pages)}")
print()
print("=" * 60)
print("TEST QUESTIONS")
print("=" * 60)
print()
print("--- IN-SCOPE (answers are in the PDF) ---")
print('1. python main.py ask "What are the three types of machine learning?"')
print('2. python main.py ask "What is the difference between precision and recall?"')
print('3. python main.py ask "Explain the bias-variance tradeoff"')
print('4. python main.py ask "What activation functions are used in neural networks?"')
print('5. python main.py ask "How does dropout prevent overfitting?"')
print('6. python main.py ask "What is the difference between batch and stochastic gradient descent?"')
print('7. python main.py ask "How do random forests improve on decision trees?"')
print('8. python main.py ask "What is the kernel trick in SVM?"')
print()
print("--- OUT-OF-SCOPE (answers are NOT in the PDF) ---")
print('9.  python main.py ask "What is a transformer architecture?"')
print('10. python main.py ask "Explain how GPT models generate text"')
print('11. python main.py ask "What is retrieval augmented generation?"')
print('12. python main.py ask "How does federated learning work?"')
print('13. python main.py ask "What is the capital of France?"')
print('14. python main.py ask "Explain quantum computing"')