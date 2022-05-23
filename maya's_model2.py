#Imports
"""

! pip install segmentation_models

# Commented out IPython magic to ensure Python compatibility.
from segmentation_models import Unet
from tensorflow import keras
from segmentation_models import get_preprocessing
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import FScore
from keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam  
import segmentation_models as sm
sm.set_framework('tf.keras')

import cv2
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
# % matplotlib online

"""#Train Model"""

from google.colab import drive
drive.mount('/content/gdrive')

os.chdir('/content/gdrive/MyDrive/tfrecords1')
os.getcwd()
path = "/content/gdrive/MyDrive/tfrecords1"

os.chdir('/content/gdrive/MyDrive/tfrecords2')
os.getcwd()
path = "/content/gdrive/MyDrive/tfrecords2"

training_filenames = os.listdir(path)
new_filenames = []
for f in training_filenames:
    new_filenames.append(os.path.join(path, f))

np.random.shuffle(new_filenames)
valid_filenames = new_filenames[:1877]
train_filenames = new_filenames[1877:

def read_tfrecord(example):
    paddings = tf.constant([[8,8],[8,8],[0,0]])
    features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string)
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
    image_bits = example["image"]
    image = tf.io.decode_png(image_bits, channels=3)
    image = tf.pad(image, paddings, "CONSTANT")
    image = tf.cast(image, tf.float32)

    label_bits = example["label"]
    label = tf.io.decode_png(label_bits, channels=1)
    label = tf.pad(label, paddings, "CONSTANT")
    label = tf.divide(label, 255)
    label = tf.where(label > 0., 1., 0.)
    return image, label

def get_training_dataset():
    return get_batched_dataset(train_filenames)

def get_valid_dataset():
    return get_batched_dataset(valid_filenames)

def get_batched_dataset(filenames):
  option_no_order = tf.data.Options()
  option_no_order.experimental_deterministic = False
  dataset = tf.data.Dataset.list_files(filenames)
  dataset = dataset.with_options(option_no_order)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
  dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

  dataset = dataset.cache() # This dataset fits in RAM
  dataset = dataset.repeat()
  dataset = dataset.shuffle(2048)
  dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 
  dataset = dataset.prefetch(AUTO) #
  
  return dataset

# define model
model = Unet(BACKBONE, encoder_weights=None)
# model.compile('Adam', loss=DiceLoss(), metrics=[FScore()])

dataset = get_training_dataset()
max=25
itr=0
for images, labels in dataset:
    for i, image in enumerate(images):
        plt.figure(figsize=(10,10))
        plt.subplot(1,3,1)
        image = np.array(image).astype(int)
        plt.imshow(image, cmap='gray')
        plt.subplot(1,3,2)
        label = labels[i]
        plt.imshow(np.reshape(label, (256, 256)), cmap='gray')
        plt.subplot(1,3,3)
        prediction = model.predict(np.reshape(image,(1,256,256,3)))
        plt.imshow(np.reshape(prediction,(256,256)), cmap='gray')
        plt.show()
    itr=itr+1
    if itr>max:
        break

dataset = get_valid_dataset()
max=25
itr=0
for images, labels in dataset:
    for i, image in enumerate(images):
        plt.figure(figsize=(10,10))
        plt.subplot(1,3,1)
        image = np.array(image).astype(int)
        plt.imshow(image, cmap='gray')
        plt.subplot(1,3,2)
        label = labels[i]
        plt.imshow(np.reshape(label, (256, 256)), cmap='gray')
        plt.subplot(1,3,3)
        prediction = model.predict(np.reshape(image,(1,256,256,3)))
        plt.imshow(np.reshape(prediction,(256,256)), cmap='gray')
        plt.show()
    itr=itr+1
    if itr>max:
        break

loss_fn = DiceLoss()
train_acc_metric = MeanIoU(2)
optimizer = Adam()
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

val_acc_metric = MeanIoU(2)
@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)
    return loss_value

"""#Run Model"""

BATCH_SIZE = 1
AUTO = tf.data.AUTOTUNE
BACKBONE = 'resnet34'

# Commented out IPython magic to ensure Python compatibility.
import time
epochs = 70
train_dataset = get_training_dataset()
val_dataset = get_valid_dataset()
steps_per_epoch = 200
val_steps=100
train_acc_value = []
val_acc_value = []
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)
        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
#                  % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * BATCH_SIZE))
        if step > steps_per_epoch:
          break
        
    
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    train_acc_value.append(train_acc)
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    # Run a validation loop at the end of each epoch.
    for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        test_loss=test_step(x_batch_val, y_batch_val)
        if step > val_steps:
            break
    print("Validation loss: %.4f" % (float(test_loss),))
    val_acc = val_acc_metric.result()
    val_acc_value.append(val_acc)
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))
