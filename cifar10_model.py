"""
cifar10_model.py
----------------
Training script for CIFAR-10 image classification using Transfer Learning (VGG16).
This script builds, trains, fine-tunes, and evaluates the model.
It generates evaluation metrics, training curves, confusion matrix, 
Grad-CAM visualizations, and sample predictions.

For a fast dry-run on CPU, set the environment variable FAST_DRY_RUN=True.
"""

# ==========================================
# 1. IMPORTS AND SETUP
# ==========================================
print("[INFO] Starting Step 1: Imports and Setup...")

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(f"[INFO] Random seed set to {RANDOM_SEED} for reproducibility.")
print(f"[INFO] TensorFlow version: {tf.__version__}")

# Check if FAST_DRY_RUN is enabled
FAST_DRY_RUN = os.getenv("FAST_DRY_RUN", "False").lower() in ("true", "1", "yes")
if FAST_DRY_RUN:
    print("[WARNING] FAST_DRY_RUN is ENABLED. The model will train on a small subset for 1 epoch to verify execution quickly.")

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[INFO] GPU detected: {gpus}. Using GPU for training.")
else:
    print("[INFO] No GPU detected. Training will run on CPU.")


# ==========================================
# 2. LOAD AND PREPROCESS DATA
# ==========================================
print("\n[INFO] Starting Step 2: Load and Preprocess Data...")

# Load CIFAR-10 dataset
(x_train_raw, y_train), (x_test_raw, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"[INFO] Loaded raw train shape: {x_train_raw.shape}, raw test shape: {x_test_raw.shape}")

# Define all 10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"[INFO] CIFAR-10 classes: {class_names}")

# Resize images from 32x32 to 48x48 in batches (VGG16 needs larger input)
def resize_dataset(images, target_size=(48, 48)):
    print(f"[INFO] Resizing images to {target_size} in batches...")
    resized_images = []
    batch_size = 5000
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        # Resize to target size
        resized_batch = tf.image.resize(batch, target_size)
        resized_images.append(resized_batch.numpy())
    return np.concatenate(resized_images, axis=0)

x_train_resized = resize_dataset(x_train_raw)
x_test_resized = resize_dataset(x_test_raw)

# Normalize pixel values to 0-1 for general/visualization purposes
x_train_norm = x_train_resized / 255.0
x_test_norm = x_test_resized / 255.0

# Apply preprocess_input from keras.applications.vgg16 to resized images
# Note: VGG16 preprocess_input expects inputs in range [0, 255] (or floats centered around ImageNet mean)
x_train_prep = preprocess_input(x_train_resized.copy())
x_test_prep = preprocess_input(x_test_resized.copy())

# If FAST_DRY_RUN is enabled, take a small subset
if FAST_DRY_RUN:
    x_train_resized = x_train_resized[:1000]
    x_train_prep = x_train_prep[:1000]
    y_train = y_train[:1000]
    x_test_norm = x_test_norm[:200]
    x_test_prep = x_test_prep[:200]
    y_test = y_test[:200]
    print(f"[INFO] Dry run subset sizes -> train: {x_train_prep.shape}, test: {x_test_prep.shape}")

print(f"[INFO] Preprocessed train shape: {x_train_prep.shape}, test shape: {x_test_prep.shape}")
print(f"[INFO] Normalized train shape (for plots): {x_train_norm.shape}")


# ==========================================
# 3. DATA AUGMENTATION
# ==========================================
print("\n[INFO] Starting Step 3: Data Augmentation...")

# Apply ImageDataGenerator on training data.
# The preprocessing_function is set to VGG16's preprocess_input.
# We will fit it on raw (0-255) images so that augmentation is done in 0-255 range, 
# then standard VGG16 mean subtraction is applied.
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    preprocessing_function=preprocess_input
)

print("[INFO] ImageDataGenerator configured with: horizontal flip, rotation 15, zoom 0.1, width/height shift 0.1, brightness [0.8, 1.2].")


# ==========================================
# 4. BUILD MODEL USING TRANSFER LEARNING
# ==========================================
print("\n[INFO] Starting Step 4: Build Model Using Transfer Learning...")

# Load VGG16 base
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze VGG16 base layers
base_model.trainable = False
print("[INFO] Loaded VGG16 base model with ImageNet weights and frozen all layers.")

# Add custom top layers using Functional API (better for layer access / Grad-CAM)
inputs = base_model.input
x = base_model.output
x = layers.GlobalAveragePooling2D(name='custom_gap')(x)
x = layers.Dense(256, activation='relu', name='custom_fc1')(x)
x = layers.BatchNormalization(name='custom_bn1')(x)
x = layers.Dropout(0.5, name='custom_dropout1')(x)
x = layers.Dense(128, activation='relu', name='custom_fc2')(x)
x = layers.Dropout(0.3, name='custom_dropout2')(x)
outputs = layers.Dense(10, activation='softmax', name='custom_predictions')(x)

model = models.Model(inputs=inputs, outputs=outputs, name='CIFAR10_VGG16_Transfer')

# Print model summary
model.summary()


# ==========================================
# 5. COMPILE THE MODEL
# ==========================================
print("\n[INFO] Starting Step 5: Compile the Model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("[INFO] Model compiled. Optimizer: Adam (lr=1e-4), Loss: sparse_categorical_crossentropy, Metrics: ['accuracy']")


# ==========================================
# 6. CALLBACKS
# ==========================================
print("\n[INFO] Starting Step 6: Setting up Callbacks...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_cifar10_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

print("[INFO] Configured EarlyStopping (patience=5), ModelCheckpoint (best_cifar10_model.h5), and ReduceLROnPlateau (factor=0.5, patience=3).")


# ==========================================
# 7. TRAIN PHASE 1 (Frozen base)
# ==========================================
print("\n[INFO] Starting Step 7: Train Phase 1 (Frozen Base)...")

epochs_p1 = 1 if FAST_DRY_RUN else 20
batch_size = 32

print(f"[INFO] Training for {epochs_p1} epochs with batch size {batch_size}...")

# Fit data using ImageDataGenerator flow
history_p1 = model.fit(
    datagen.flow(x_train_resized, y_train, batch_size=batch_size),
    epochs=epochs_p1,
    validation_data=(x_test_prep, y_test),
    callbacks=callbacks,
    verbose=1
)


# ==========================================
# 8. FINE TUNING — TRAIN PHASE 2
# ==========================================
print("\n[INFO] Starting Step 8: Fine-Tuning - Train Phase 2...")

# Unfreeze last 4 layers of VGG16
# Let's inspect base_model layers and unfreeze the last 4.
# VGG16 base layers count: 19 layers
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

print("[INFO] Unfrozen the last 4 layers of VGG16 base model for fine-tuning:")
for idx, layer in enumerate(base_model.layers):
    if layer.trainable:
        print(f"  - Layer {idx}: {layer.name} (TRAINABLE)")

# Recompile with very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("[INFO] Model recompiled with lower learning rate: Adam (lr=1e-5).")

epochs_p2 = 1 if FAST_DRY_RUN else 10
print(f"[INFO] Training Phase 2 (Fine-tuning) for {epochs_p2} more epochs...")

history_p2 = model.fit(
    datagen.flow(x_train_resized, y_train, batch_size=batch_size),
    epochs=epochs_p1 + epochs_p2,  # fit continues from current epoch
    initial_epoch=epochs_p1,
    validation_data=(x_test_prep, y_test),
    callbacks=callbacks,
    verbose=1
)


# ==========================================
# 9. PLOT TRAINING RESULTS
# ==========================================
print("\n[INFO] Starting Step 9: Plot Training Results...")

# Combine history from both phases
acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
loss = history_p1.history['loss'] + history_p2.history['loss']
val_loss = history_p1.history['val_loss'] + history_p2.history['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='#1f77b4', linewidth=2.5)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2.5)
plt.axvline(x=epochs_p1, color='#d62728', linestyle='--', linewidth=1.5, label='Fine-Tuning Start')
plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='#1f77b4', linewidth=2.5)
plt.plot(epochs_range, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2.5)
plt.axvline(x=epochs_p1, color='#d62728', linestyle='--', linewidth=1.5, label='Fine-Tuning Start')
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300)
plt.close()
print("[INFO] Combined training accuracy/loss plot saved as 'training_results.png'.")


# ==========================================
# 10. EVALUATE MODEL
# ==========================================
print("\n[INFO] Starting Step 10: Evaluate Model...")

test_loss, test_acc = model.evaluate(x_test_prep, y_test, verbose=0)
print("\n" + "="*50)
print("             FINAL MODEL TEST RESULTS")
print("="*50)
print(f"Test Loss:            {test_loss:.4f}")
print(f"Test Accuracy:        {test_acc:.4f}")
print(f"Test Accuracy (%):    {test_acc * 100:.2f}%")
print("="*50 + "\n")


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================
print("[INFO] Starting Step 11: Confusion Matrix heatmap...")

# Get predictions
y_pred_probs = model.predict(x_test_prep)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on CIFAR-10 Test Set', fontsize=14, fontweight='bold')
plt.ylabel('Actual Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("[INFO] Confusion Matrix heatmap saved as 'confusion_matrix.png'.")


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================
print("\n[INFO] Starting Step 12: Classification Report...")

report = classification_report(y_test, y_pred, target_names=class_names)
print("Classification Report:")
print(report)


# ==========================================
# 13. GRAD-CAM VISUALIZATION
# ==========================================
print("\n[INFO] Starting Step 13: Grad-CAM Visualization...")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes Grad-CAM heatmap for a given image array and model.
    """
    # Create a sub-model that outputs the feature maps of the last conv layer and the final predictions
    grad_model = models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients of the class score w.r.t. last conv feature maps
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Global average pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by its weight, then sum
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep positive activations, and normalize
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap).numpy()
    if max_val == 0:
        max_val = 1e-10
    heatmap = heatmap / max_val
    return heatmap.numpy()

def superimpose_gradcam(img_norm, heatmap, alpha=0.5):
    """
    Superimposes the Grad-CAM heatmap onto the normalized image.
    """
    # Resize heatmap to match image size (48, 48) using tf.image.resize
    heatmap_expanded = np.expand_dims(heatmap, axis=-1)
    heatmap_resized = tf.image.resize(heatmap_expanded, (img_norm.shape[0], img_norm.shape[1])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)
    
    # Scale to 0-255
    heatmap_255 = np.uint8(255 * heatmap_resized)
    
    # Using matplotlib to colorize the heatmap
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]  # Extract RGB channels
    jet_heatmap = jet_colors[heatmap_255]    # Map heatmap values to jet color space
    
    # Superimpose heatmap and original image
    superimposed_img = jet_heatmap * alpha + img_norm * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0.0, 1.0)
    return superimposed_img

# Generate Grad-CAM for 5 test images
fig, axes = plt.subplots(5, 2, figsize=(8, 18))
for i in range(5):
    img_norm = x_test_norm[i]
    img_prep = x_test_prep[i:i+1]
    
    actual_class = class_names[y_test[i][0]]
    pred_idx = y_pred[i]
    pred_class = class_names[pred_idx]
    confidence = y_pred_probs[i][pred_idx] * 100
    
    # Calculate Grad-CAM Heatmap
    heatmap = make_gradcam_heatmap(img_prep, model, 'block5_conv3')
    superimposed = superimpose_gradcam(img_norm, heatmap, alpha=0.4)
    
    # Plot original image
    axes[i, 0].imshow(img_norm)
    axes[i, 0].set_title(f"Original: {actual_class}", fontsize=11, fontweight='bold')
    axes[i, 0].axis('off')
    
    # Plot Grad-CAM
    axes[i, 1].imshow(superimposed)
    axes[i, 1].set_title(f"Grad-CAM (Pred: {pred_class} {confidence:.1f}%)", fontsize=11, fontweight='bold')
    axes[i, 1].axis('off')

plt.suptitle("Grad-CAM Activation Heatmaps (VGG16 Last Conv Layer)", fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('gradcam_results.png', dpi=300)
plt.close()
print("[INFO] Grad-CAM results visualization saved as 'gradcam_results.png'.")


# ==========================================
# 14. SAMPLE PREDICTIONS
# ==========================================
print("\n[INFO] Starting Step 14: Sample Predictions grid...")

fig = plt.figure(figsize=(14, 7))
random_indices = np.random.choice(len(x_test_norm), 10, replace=False)

for idx, test_idx in enumerate(random_indices):
    # Retrieve normalized image for display
    img_norm = x_test_norm[test_idx]
    
    # Retrieve preprocessed image batch for prediction
    img_prep = x_test_prep[test_idx:test_idx+1]
    
    probs = model.predict(img_prep, verbose=0)
    pred_idx = np.argmax(probs[0])
    actual_idx = y_test[test_idx][0]
    
    pred_class = class_names[pred_idx]
    actual_class = class_names[actual_idx]
    confidence = probs[0][pred_idx] * 100
    
    correct = (pred_idx == actual_idx)
    color = 'green' if correct else 'red'
    title_text = f"Pred: {pred_class} ({confidence:.1f}%)\nAct: {actual_class}"
    
    plt.subplot(2, 5, idx + 1)
    plt.imshow(img_norm)
    plt.title(title_text, color=color, fontsize=10, fontweight='bold')
    plt.axis('off')

plt.suptitle("Model Predictions on Sample CIFAR-10 Images\n(Green = Correct, Red = Incorrect)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('predictions.png', dpi=300)
plt.close()
print("[INFO] Sample predictions grid saved as 'predictions.png'.")


# ==========================================
# 15. SAVE MODEL
# ==========================================
print("\n[INFO] Starting Step 15: Save Model...")

# Save final model
model.save('cifar10_vgg16_model.h5')
print("[INFO] Saved final model as 'cifar10_vgg16_model.h5'.")

# Print all output files
print("\n" + "="*50)
print("             ALL GENERATED OUTPUT FILES")
print("="*50)
print("1. best_cifar10_model.h5   -> Best model checkpoints (val_accuracy based)")
print("2. cifar10_vgg16_model.h5  -> Final trained & fine-tuned model")
print("3. training_results.png    -> Loss & Accuracy curve plot")
print("4. confusion_matrix.png   -> Heatmap of model predictions vs targets")
print("5. gradcam_results.png     -> Grad-CAM visualizations on test images")
print("6. predictions.png         -> 2x5 grid of sample predictions and results")
print("="*50 + "\n")
print("[INFO] Model training process complete!")
