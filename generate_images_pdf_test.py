"""
Generate a test PDF with embedded diagrams/charts for image handling testing.
Uses PyMuPDF to draw shapes, charts, and diagrams directly into the PDF.
Run: python generate_image_test_pdf.py
"""

import fitz
import os

os.makedirs("data/raw_pdfs", exist_ok=True)

doc = fitz.open()


# ─────────────────────────────────────────────
# PAGE 1: Neural Network Architecture Diagram
# ─────────────────────────────────────────────
page = doc.new_page(width=612, height=792)

# Title
title_rect = fitz.Rect(50, 30, 562, 60)
page.insert_textbox(title_rect, "Neural Network Architecture", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.5))
page.draw_line(fitz.Point(50, 65), fitz.Point(562, 65), color=(0.3, 0.3, 0.3), width=0.5)

# Text above diagram
text_rect = fitz.Rect(50, 75, 562, 160)
page.insert_textbox(text_rect, (
    "A feedforward neural network consists of an input layer, one or more hidden layers, and an output layer. "
    "Each neuron in one layer connects to every neuron in the next layer (fully connected). "
    "The network below shows a 3-2-1 architecture: 3 input neurons, 2 hidden neurons, and 1 output neuron."
), fontsize=10, fontname="helv")

# Draw neural network diagram
# Input layer (3 neurons)
input_positions = [(150, 250), (150, 350), (150, 450)]
hidden_positions = [(350, 300), (350, 400)]
output_positions = [(500, 350)]

# Draw connections first (behind neurons)
for ix, iy in input_positions:
    for hx, hy in hidden_positions:
        page.draw_line(fitz.Point(ix + 15, iy), fitz.Point(hx - 15, hy), color=(0.7, 0.7, 0.7), width=0.8)

for hx, hy in hidden_positions:
    for ox, oy in output_positions:
        page.draw_line(fitz.Point(hx + 15, hy), fitz.Point(ox - 15, oy), color=(0.7, 0.7, 0.7), width=0.8)

# Draw neurons
for x, y in input_positions:
    page.draw_circle(fitz.Point(x, y), 15, color=(0.2, 0.4, 0.8), fill=(0.3, 0.5, 0.9), width=1.5)

for x, y in hidden_positions:
    page.draw_circle(fitz.Point(x, y), 15, color=(0.2, 0.7, 0.3), fill=(0.3, 0.8, 0.4), width=1.5)

for x, y in output_positions:
    page.draw_circle(fitz.Point(x, y), 15, color=(0.8, 0.2, 0.2), fill=(0.9, 0.3, 0.3), width=1.5)

# Layer labels
page.insert_text(fitz.Point(120, 510), "Input Layer", fontsize=10, fontname="helv", color=(0.2, 0.4, 0.8))
page.insert_text(fitz.Point(310, 460), "Hidden Layer", fontsize=10, fontname="helv", color=(0.2, 0.7, 0.3))
page.insert_text(fitz.Point(465, 400), "Output Layer", fontsize=10, fontname="helv", color=(0.8, 0.2, 0.2))

# Neuron labels
for i, (x, y) in enumerate(input_positions):
    page.insert_text(fitz.Point(x - 5, y + 4), f"x{i+1}", fontsize=8, fontname="helv", color=(1, 1, 1))
for i, (x, y) in enumerate(hidden_positions):
    page.insert_text(fitz.Point(x - 5, y + 4), f"h{i+1}", fontsize=8, fontname="helv", color=(1, 1, 1))
page.insert_text(fitz.Point(output_positions[0][0] - 3, output_positions[0][1] + 4), "y", fontsize=8, fontname="helv", color=(1, 1, 1))

# Weight labels on some connections
page.insert_text(fitz.Point(230, 260), "w11", fontsize=7, fontname="helv", color=(0.4, 0.4, 0.4))
page.insert_text(fitz.Point(240, 320), "w12", fontsize=7, fontname="helv", color=(0.4, 0.4, 0.4))
page.insert_text(fitz.Point(420, 320), "w'1", fontsize=7, fontname="helv", color=(0.4, 0.4, 0.4))

# Caption
caption_rect = fitz.Rect(50, 530, 562, 570)
page.insert_textbox(caption_rect, "Figure 1: A fully connected feedforward neural network with 3 inputs, 2 hidden neurons, and 1 output.",
                     fontsize=9, fontname="helv", color=(0.3, 0.3, 0.3))

# Text below diagram
text2_rect = fitz.Rect(50, 580, 562, 740)
page.insert_textbox(text2_rect, (
    "In this architecture, each connection carries a weight (w). During forward propagation, "
    "the input values are multiplied by their weights, summed at each hidden neuron, and passed through "
    "an activation function (typically ReLU). The process repeats from hidden to output layer. "
    "The output neuron produces the final prediction. During training, backpropagation adjusts all weights "
    "to minimize the loss function. The number of parameters in this network is (3x2) + 2 + (2x1) + 1 = 11, "
    "where the extra terms account for bias values at each neuron."
), fontsize=10, fontname="helv")


# ─────────────────────────────────────────────
# PAGE 2: Training Loss Curve
# ─────────────────────────────────────────────
page2 = doc.new_page(width=612, height=792)

title_rect = fitz.Rect(50, 30, 562, 60)
page2.insert_textbox(title_rect, "Training vs Validation Loss", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.5))
page2.draw_line(fitz.Point(50, 65), fitz.Point(562, 65), color=(0.3, 0.3, 0.3), width=0.5)

text_rect = fitz.Rect(50, 75, 562, 140)
page2.insert_textbox(text_rect, (
    "The plot below shows training loss and validation loss over 50 epochs. "
    "Both curves decrease initially, but after epoch 25 the validation loss begins to increase "
    "while training loss continues to decrease. This divergence indicates overfitting."
), fontsize=10, fontname="helv")

# Draw axes
origin_x, origin_y = 100, 500
end_x, end_y = 500, 500
top_y = 180

page2.draw_line(fitz.Point(origin_x, origin_y), fitz.Point(end_x, end_y), color=(0, 0, 0), width=1.5)
page2.draw_line(fitz.Point(origin_x, origin_y), fitz.Point(origin_x, top_y), color=(0, 0, 0), width=1.5)

# Axis labels
page2.insert_text(fitz.Point(280, 530), "Epochs", fontsize=10, fontname="helv")
page2.insert_text(fitz.Point(55, 340), "Loss", fontsize=10, fontname="helv")

# Tick marks on x-axis
for i in range(0, 6):
    x = origin_x + i * 80
    page2.draw_line(fitz.Point(x, origin_y), fitz.Point(x, origin_y + 5), color=(0, 0, 0), width=1)
    page2.insert_text(fitz.Point(x - 5, origin_y + 18), str(i * 10), fontsize=8, fontname="helv")

# Tick marks on y-axis
for i in range(0, 5):
    y = origin_y - i * 80
    page2.draw_line(fitz.Point(origin_x - 5, y), fitz.Point(origin_x, y), color=(0, 0, 0), width=1)
    label = f"{i * 0.25:.2f}"
    page2.insert_text(fitz.Point(origin_x - 35, y + 4), label, fontsize=8, fontname="helv")

# Training loss curve (steadily decreasing)
import math
train_points = []
for i in range(50):
    x = origin_x + (i / 50) * 400
    loss = 0.9 * math.exp(-0.06 * i) + 0.05
    y = origin_y - loss * 320
    train_points.append(fitz.Point(x, y))

for i in range(len(train_points) - 1):
    page2.draw_line(train_points[i], train_points[i + 1], color=(0.2, 0.4, 0.9), width=2)

# Validation loss curve (decreases then increases - overfitting)
val_points = []
for i in range(50):
    x = origin_x + (i / 50) * 400
    loss = 0.95 * math.exp(-0.05 * i) + 0.08 + max(0, (i - 25) * 0.008)
    y = origin_y - loss * 320
    val_points.append(fitz.Point(x, y))

for i in range(len(val_points) - 1):
    page2.draw_line(val_points[i], val_points[i + 1], color=(0.9, 0.3, 0.2), width=2)

# Legend
page2.draw_line(fitz.Point(380, 170), fitz.Point(410, 170), color=(0.2, 0.4, 0.9), width=2)
page2.insert_text(fitz.Point(415, 174), "Training Loss", fontsize=9, fontname="helv")
page2.draw_line(fitz.Point(380, 190), fitz.Point(410, 190), color=(0.9, 0.3, 0.2), width=2)
page2.insert_text(fitz.Point(415, 194), "Validation Loss", fontsize=9, fontname="helv")

# Overfitting annotation
page2.insert_text(fitz.Point(350, 260), "Overfitting", fontsize=10, fontname="helv", color=(0.8, 0, 0))
page2.draw_line(fitz.Point(340, 290), fitz.Point(340, 500), color=(0.8, 0, 0), width=1, dashes="[4 4]")
page2.insert_text(fitz.Point(320, 515), "Epoch 25", fontsize=8, fontname="helv", color=(0.8, 0, 0))

# Caption
caption_rect = fitz.Rect(50, 550, 562, 590)
page2.insert_textbox(caption_rect, "Figure 2: Training and validation loss curves showing overfitting after epoch 25.",
                      fontsize=9, fontname="helv", color=(0.3, 0.3, 0.3))

text2_rect = fitz.Rect(50, 600, 562, 740)
page2.insert_textbox(text2_rect, (
    "Early stopping would halt training at approximately epoch 25, where validation loss is at its minimum. "
    "This is the point where the model has learned the underlying patterns without memorizing noise. "
    "The gap between training and validation loss after this point is called the generalization gap. "
    "A larger gap indicates more severe overfitting. Techniques like dropout, L2 regularization, and "
    "data augmentation can help reduce this gap and extend the useful training window."
), fontsize=10, fontname="helv")


# ─────────────────────────────────────────────
# PAGE 3: Confusion Matrix
# ─────────────────────────────────────────────
page3 = doc.new_page(width=612, height=792)

title_rect = fitz.Rect(50, 30, 562, 60)
page3.insert_textbox(title_rect, "Confusion Matrix for Binary Classification", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.5))
page3.draw_line(fitz.Point(50, 65), fitz.Point(562, 65), color=(0.3, 0.3, 0.3), width=0.5)

text_rect = fitz.Rect(50, 75, 562, 160)
page3.insert_textbox(text_rect, (
    "A confusion matrix summarizes the performance of a classification model by showing the counts of "
    "true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). "
    "The matrix below shows results from a spam email classifier tested on 200 emails."
), fontsize=10, fontname="helv")

# Draw confusion matrix
cm_x, cm_y = 160, 200
cell_w, cell_h = 130, 100

# Header labels
page3.insert_text(fitz.Point(cm_x + 60, cm_y - 25), "Predicted", fontsize=12, fontname="helv", color=(0.2, 0.2, 0.2))
page3.insert_text(fitz.Point(cm_x + cell_w // 2 - 10, cm_y - 8), "Spam", fontsize=10, fontname="helv")
page3.insert_text(fitz.Point(cm_x + cell_w + cell_w // 2 - 15, cm_y - 8), "Not Spam", fontsize=10, fontname="helv")

page3.insert_text(fitz.Point(cm_x - 60, cm_y + cell_h + 10), "Actual", fontsize=12, fontname="helv", color=(0.2, 0.2, 0.2))
page3.insert_text(fitz.Point(cm_x - 45, cm_y + cell_h // 2 + 5), "Spam", fontsize=10, fontname="helv")
page3.insert_text(fitz.Point(cm_x - 60, cm_y + cell_h + cell_h // 2 + 5), "Not Spam", fontsize=10, fontname="helv")

# Cells
matrix_data = [
    [("TP = 85", (0.2, 0.7, 0.3, 0.3)), ("FN = 15", (0.9, 0.3, 0.3, 0.3))],
    [("FP = 10", (0.9, 0.6, 0.2, 0.3)), ("TN = 90", (0.2, 0.5, 0.8, 0.3))],
]

for row in range(2):
    for col in range(2):
        x = cm_x + col * cell_w
        y = cm_y + row * cell_h
        rect = fitz.Rect(x, y, x + cell_w, y + cell_h)
        label, fill_color = matrix_data[row][col]

        page3.draw_rect(rect, color=(0.3, 0.3, 0.3), fill=fill_color, width=1.5)

        # Center the text in cell
        text_x = x + cell_w // 2 - 20
        text_y = y + cell_h // 2 + 5
        page3.insert_text(fitz.Point(text_x, text_y), label, fontsize=14, fontname="helv", color=(0, 0, 0))

# Caption
caption_rect = fitz.Rect(50, cm_y + 2 * cell_h + 20, 562, cm_y + 2 * cell_h + 50)
page3.insert_textbox(caption_rect, "Figure 3: Confusion matrix for spam classifier (200 test emails).",
                      fontsize=9, fontname="helv", color=(0.3, 0.3, 0.3))

# Metrics calculation
metrics_rect = fitz.Rect(50, cm_y + 2 * cell_h + 60, 562, 740)
page3.insert_textbox(metrics_rect, (
    "From this confusion matrix, we can calculate the following metrics:\n\n"
    "Accuracy = (TP + TN) / Total = (85 + 90) / 200 = 87.5%\n"
    "Precision = TP / (TP + FP) = 85 / (85 + 10) = 89.5%\n"
    "Recall = TP / (TP + FN) = 85 / (85 + 15) = 85.0%\n"
    "F1 Score = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (0.895 * 0.85) / (0.895 + 0.85) = 87.2%\n\n"
    "The classifier has high precision (few false spam flags) but lower recall (misses 15% of actual spam). "
    "This tradeoff depends on the application: for spam filtering, higher recall might be preferred to catch more spam, "
    "even at the cost of occasionally flagging legitimate emails."
), fontsize=10, fontname="helv")


# ─────────────────────────────────────────────
# PAGE 4: Decision Boundary Visualization
# ─────────────────────────────────────────────
page4 = doc.new_page(width=612, height=792)

title_rect = fitz.Rect(50, 30, 562, 60)
page4.insert_textbox(title_rect, "Decision Boundaries: Underfitting vs Overfitting", fontsize=18, fontname="helv", color=(0.1, 0.1, 0.5))
page4.draw_line(fitz.Point(50, 65), fitz.Point(562, 65), color=(0.3, 0.3, 0.3), width=0.5)

text_rect = fitz.Rect(50, 75, 562, 140)
page4.insert_textbox(text_rect, (
    "The three plots below show how model complexity affects the decision boundary for a binary "
    "classification task. Blue circles represent Class A and red squares represent Class B."
), fontsize=10, fontname="helv")

import random
random.seed(42)

# Generate data points for all three plots
class_a = [(random.gauss(0.35, 0.12), random.gauss(0.55, 0.12)) for _ in range(15)]
class_b = [(random.gauss(0.65, 0.12), random.gauss(0.45, 0.12)) for _ in range(15)]

def draw_scatter(page, offset_x, offset_y, plot_w, plot_h, title_text):
    """Draw a scatter plot with data points."""
    # Plot border
    page.draw_rect(fitz.Rect(offset_x, offset_y, offset_x + plot_w, offset_y + plot_h),
                   color=(0.3, 0.3, 0.3), width=1)

    # Title
    page.insert_text(fitz.Point(offset_x + plot_w // 2 - 30, offset_y - 8), title_text,
                     fontsize=10, fontname="helv", color=(0.2, 0.2, 0.2))

    # Data points
    for ax, ay in class_a:
        px = offset_x + ax * plot_w
        py = offset_y + (1 - ay) * plot_h
        page.draw_circle(fitz.Point(px, py), 4, color=(0.2, 0.4, 0.9), fill=(0.3, 0.5, 0.9), width=1)

    for bx, by in class_b:
        px = offset_x + bx * plot_w
        py = offset_y + (1 - by) * plot_h
        page.draw_rect(fitz.Rect(px - 4, py - 4, px + 4, py + 4),
                       color=(0.9, 0.2, 0.2), fill=(0.9, 0.3, 0.3), width=1)

# Plot 1: Underfitting (straight line, bad fit)
draw_scatter(page4, 50, 170, 150, 150, "Underfitting")
page4.draw_line(fitz.Point(50, 320), fitz.Point(200, 170), color=(0.5, 0.5, 0.5), width=2, dashes="[4 4]")

# Plot 2: Good fit (smooth curve)
draw_scatter(page4, 230, 170, 150, 150, "Good Fit")
# Draw a smooth diagonal boundary
points = [(230, 280), (270, 260), (310, 245), (350, 220), (380, 200)]
for i in range(len(points) - 1):
    page4.draw_line(fitz.Point(*points[i]), fitz.Point(*points[i + 1]), color=(0.2, 0.7, 0.2), width=2)

# Plot 3: Overfitting (jagged boundary)
draw_scatter(page4, 410, 170, 150, 150, "Overfitting")
jagged = [(410, 300), (420, 260), (440, 280), (455, 230), (470, 260), (490, 220),
          (510, 250), (530, 200), (545, 230), (560, 185)]
for i in range(len(jagged) - 1):
    page4.draw_line(fitz.Point(*jagged[i]), fitz.Point(*jagged[i + 1]), color=(0.9, 0.2, 0.2), width=2)

# Legend
page4.draw_circle(fitz.Point(80, 360), 4, color=(0.2, 0.4, 0.9), fill=(0.3, 0.5, 0.9), width=1)
page4.insert_text(fitz.Point(90, 364), "Class A", fontsize=9, fontname="helv")
page4.draw_rect(fitz.Rect(156, 356, 164, 364), color=(0.9, 0.2, 0.2), fill=(0.9, 0.3, 0.3), width=1)
page4.insert_text(fitz.Point(170, 364), "Class B", fontsize=9, fontname="helv")

# Caption
caption_rect = fitz.Rect(50, 380, 562, 410)
page4.insert_textbox(caption_rect,
    "Figure 4: Decision boundaries for underfitting (linear), good fit (smooth curve), and overfitting (jagged curve).",
    fontsize=9, fontname="helv", color=(0.3, 0.3, 0.3))

# Text
text2_rect = fitz.Rect(50, 420, 562, 740)
page4.insert_textbox(text2_rect, (
    "The left plot shows underfitting: the linear decision boundary is too simple to separate the classes. "
    "Many points from both classes fall on the wrong side. This corresponds to high bias.\n\n"
    "The center plot shows a good fit: the smooth curved boundary captures the true separation between classes "
    "without following individual noise points. This is the ideal tradeoff between bias and variance.\n\n"
    "The right plot shows overfitting: the jagged boundary perfectly separates all training points but follows "
    "noise rather than the true pattern. It will likely misclassify new data points. This corresponds to high variance.\n\n"
    "Model complexity increases from left to right. A linear model (left) has few parameters and high bias. "
    "A polynomial model (center) with moderate degree has balanced complexity. A very high-degree polynomial "
    "or a deep neural network without regularization (right) can overfit by having too many parameters relative "
    "to the amount of training data.\n\n"
    "The key insight is that training accuracy alone is misleading. The underfitting model has low training accuracy. "
    "The overfitting model has perfect training accuracy. But the good fit model will have the best test accuracy "
    "because it generalizes to unseen data."
), fontsize=10, fontname="helv")


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
output_path = "data/raw_pdfs/ml_visual_guide.pdf"
doc.save(output_path)
doc.close()

print(f"PDF generated: {output_path}")
print(f"Pages: 4")
print()
print("=" * 60)
print("TEST QUESTIONS — IMAGE-RELATED")
print("=" * 60)
print()
print("--- Questions about the diagrams ---")
print('python main.py ask "How many neurons are in the hidden layer of the neural network?"')
print('python main.py ask "What happens to validation loss after epoch 25?"')
print('python main.py ask "What is the precision of the spam classifier?"')
print('python main.py ask "What is the recall of the spam classifier?"')
print('python main.py ask "How many true positives did the spam classifier have?"')
print('python main.py ask "What does the overfitting decision boundary look like?"')
print()
print("--- Follow-up questions for chat mode ---")
print('# Start with: "Explain the confusion matrix from the spam classifier"')
print('# Then ask: "What would happen if we optimized for recall instead?"')
print('# Then ask: "How does that relate to the bias-variance tradeoff?"')