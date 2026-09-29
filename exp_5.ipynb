# ============================================================
# Experiment 5: Simple CNN for MNIST Digit Classification
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense


# ------------------------------------------------------------
# 1. Load MNIST Dataset
# ------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data()


# ------------------------------------------------------------
# 2. Preprocess the Images
# ------------------------------------------------------------

# Normalize pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension
# From (60000, 28, 28)
# To   (60000, 28, 28, 1)

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]


# ------------------------------------------------------------
# 3. Display One Sample Image
# ------------------------------------------------------------

plt.imshow(x_train[0].squeeze(), cmap="gray")
plt.title(f"Actual Digit: {y_train[0]}")
plt.axis("off")
plt.show()


# ------------------------------------------------------------
# 4. Build the CNN
# ------------------------------------------------------------

model = Sequential([

    Input(shape=(28, 28, 1)),

    # Convolution layer
    Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    # Pooling layer
    MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps into a vector
    Flatten(),

    # Fully connected layer
    Dense(
        64,
        activation="relu"
    ),

    # Output layer: 10 digits
    Dense(
        10,
        activation="softmax"
    )
])


# ------------------------------------------------------------
# 5. Display CNN Architecture
# ------------------------------------------------------------

model.summary()


# ------------------------------------------------------------
# 6. Compile the CNN
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 7. Train the CNN
# ------------------------------------------------------------

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)


# ------------------------------------------------------------
# 8. Evaluate the CNN
# ------------------------------------------------------------

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nTest Loss     :", loss)
print("Test Accuracy :", accuracy)


# ------------------------------------------------------------
# 9. Predict a Test Image
# ------------------------------------------------------------

index = 0

prediction = model.predict(
    x_test[index:index + 1],
    verbose=0
)

predicted_digit = np.argmax(prediction)

print("\nActual Digit    :", y_test[index])
print("Predicted Digit :", predicted_digit)


# ------------------------------------------------------------
# 10. Display the Prediction
# ------------------------------------------------------------

plt.imshow(
    x_test[index].squeeze(),
    cmap="gray"
)

plt.title(
    f"Actual: {y_test[index]} | "
    f"Predicted: {predicted_digit}"
)

plt.axis("off")
plt.show()
