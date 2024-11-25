import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load and resize images to 128x128 pixels
def load_images(folder_path):
    images = []
    for file_name in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, file_name))
        if image is not None:
            image = cv2.resize(image, (128, 128))
            images.append(image)
    return np.array(images)

# Load labels from CSV
def load_labels(csv_path):
    labels = pd.read_csv(csv_path)
    return labels

# Generate ground truth labels with Gaussian dilation
def generate_labels(image_shape, line_coords, sigma=4):
    label = np.zeros(image_shape[:2], dtype=np.float32)
    x1, y1, x2, y2 = line_coords
    cv2.line(label, (x1, y1), (x2, y2), color=1.0, thickness=1)
    return cv2.GaussianBlur(label, (0, 0), sigma)

# Load images and labels
images = load_images(r'C:\Users\mcoyl\OneDrive\Desktop\CMU_Robo_Capstone\dataset\images')
labels_df = load_labels(r'C:\Users\mcoyl\OneDrive\Desktop\CMU_Robo_Capstone\dataset\values.csv')

# Generate ground truth labels
labels = []
for index, row in labels_df.iterrows():
    line_coords = (row['P1x'], row['P1y'], row['P2x'], row['P2y'])
    label = generate_labels(images[0].shape, line_coords)
    labels.append(label)
labels = np.array(labels)

datagen = ImageDataGenerator(rotation_range=360, fill_mode='nearest')

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

def create_model(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='linear', input_shape=input_shape, padding='same'))
    model.add(PReLU())
    model.add(MaxPooling2D((2, 2)))
    
    for _ in range(3):
        model.add(Conv2D(16, (3, 3), activation='linear', padding='same'))
        model.add(PReLU())
    
    model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='linear'))
    model.add(PReLU())
    
    model.add(Conv2D(1, (1, 1), activation='relu', padding='same'))
    
    return model

model = create_model()
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

model.fit(
    X_train, y_train,
    validation_split=0.1,
    batch_size=128,
    epochs=25,
    callbacks=[early_stopping, reduce_lr]
)

predictions = model.predict(X_test)

def apply_threshold(prediction, tau_factor=0.35):
    tau = tau_factor * prediction.max()
    return (prediction >= tau).astype(np.uint8)

def hough_transform(thresholded_image):
    lines = cv2.HoughLines(thresholded_image, 1, np.pi/180, threshold=50)
    return lines

def find_grip_point(line, prediction):
    max_prob = 0
    grip_point = (0, 0)
    for l in line:
        rho, theta = l[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        for x, y in zip(range(x1, x2), range(y1, y2)):
            if 0 <= x < prediction.shape[1] and 0 <= y < prediction.shape[0] and prediction[y, x] > max_prob:
                max_prob = prediction[y, x]
                grip_point = (x, y)
    return grip_point

# Apply threshold and Hough transform to predictions
thresholded_predictions = [apply_threshold(pred) for pred in predictions]
lines = [hough_transform(thresh) for thresh in thresholded_predictions]

# Find grip points
grip_points = [find_grip_point(line, pred) for line, pred in zip(lines, predictions) if line is not None]

from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score

# Calculate regression metrics
mse = mean_squared_error(y_test.flatten(), predictions.flatten())
mae = mean_absolute_error(y_test.flatten(), predictions.flatten())

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Binarize the ground truth and predictions for classification metrics
y_test_binary = (y_test.flatten() > 0.5).astype(int)
predictions_binary = (predictions.flatten() > 0.5).astype(int)

# Calculate classification metrics
precision = precision_score(y_test_binary, predictions_binary)
recall = recall_score(y_test_binary, predictions_binary)
f1 = f1_score(y_test_binary, predictions_binary)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Visualize one of the test images with the predicted line and grip point
plt.imshow(X_test[0])
if lines[0] is not None:
    for l in lines[0]:
        rho, theta = l[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        plt.plot([x1, x2], [y1, y2], color='red')
plt.scatter(grip_points[0][0], grip_points[0][1], color='blue')
plt.show()