# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import librosa
import librosa.display
import glob
import skimage
import seaborn as sns
import skimage
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

!pip install resampy

from google.colab import drive
drive.mount('/content/drive')

"""**EDA**"""

import IPython.display as ipd
filepath = "/content/drive/MyDrive/ds/fold1/101415-3-0-2.wav"
data, sample_rate = librosa.load(filepath)
plt.figure(figsize=(12, 5))
librosa.display.waveshow(data, sr=sample_rate)
ipd.Audio(filepath)

"""**IMBALANCE DATASET CHECK**"""

df = pd.read_csv("/content/drive/MyDrive/ds/UrbanSound8K.csv")

df.head()

df['class'].value_counts()

import seaborn as sns
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=df, orient='v', palette='Set1')
plt.title("Count of records in each class")
plt.xticks(rotation="vertical")
plt.show()

"""**DATA PREPROCESSING**"""

# APPLYING MFCC ON A SINGLE AUDIO FILE

"""**MFCC'S**"""

1
print(mfccs.shape)
print(mfccs)

def features_extractor(file):
    #load the file (audio)
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    #we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

!pip install resampy
#Now we ned to extract the featured from all the audio files so we use tqdm
import numpy as np
from tqdm import tqdm
import os
### Now we iterate through every audio file and extract features
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for index_num,row in tqdm(df.iterrows()):
    file_name = os.path.join(os.path.abspath('/content/drive/MyDrive/ds'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()

"""**TRAIN TEST SPLIT**"""

#First, we split the dependent and independent features.
#After that, we have 10 classes, so we use label encoding(Integer label encoding) from number 1 to 10 and
#convert it into categories. After that, we split the data into train and test sets in an 80-20 ratio.

### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

### Label Encoding -> Label Encoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

# Define the number of classes
num_classes = y.shape[1]

print(y)

### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

X_train.shape[0], X_test.shape[0]

# Reshape the input data to add a channel dimension
X_train = X_train.reshape((*X_train.shape, 1))
X_test = X_test.reshape((*X_test.shape, 1))

# Check the shape after reshaping
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

"""**Environmental Sound Classification Model Creation**

**STACKED CNN**
"""

# Define the stacked CNN model

input_shape = X_train.shape[1:]

# Define a function to schedule the learning rate
def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.9  # You can adjust the factor by which you want to decrease the learning rate
    return lr

# Create a learning rate scheduler callback
lr_scheduler = LearningRateScheduler(scheduler)


#3 convolutinal layers with increasing number of filters. got 92%
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 1)),  # Adjust pool_size to (2, 1)
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])



# Compile the model with an initial learning rate
initial_learning_rate = 0.001  # Initial learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model with learning rate scheduler
history = model.fit(X_train, y_train, epochs=90, batch_size=64, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

"""**CNN ARCHITECTURE**"""

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='cnn_architecture.png', show_shapes=True)

import tensorflow.keras as keras
import networkx as nx
import matplotlib.pyplot as plt

def visualize_model(model, filename='model.png'):
    G = nx.DiGraph()

    # Add nodes for layers
    for layer in model.layers:
        G.add_node(layer.name, label=layer.name)

    # Add edges between layers
    for layer in model.layers:
        inbound_nodes = getattr(layer, '_inbound_nodes', None)
        if inbound_nodes:
            for node in inbound_nodes:
                for inbound_layer in node.inbound_layers:
                    G.add_edge(inbound_layer.name, layer.name)

    pos = nx.spring_layout(G)  # Position nodes using spring layout

    # Draw nodes with custom shapes and colors
    shapes = {
        'InputLayer': 's',
        'Conv2D': 'o',
        'MaxPooling2D': 'd',
        'Dense': 'p'
    }
    colors = {
        'InputLayer': 'lightgray',
        'Conv2D': 'lightblue',
        'MaxPooling2D': 'lightgreen',
        'Dense': 'lightcoral'
    }

    for node in G.nodes:
        node_type = model.get_layer(node).__class__.__name__
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape=shapes.get(node_type, 'o'), node_color=colors.get(node_type, 'white'), node_size=500)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Add legend
    legend_handles = []
    for node_type, shape in shapes.items():
        legend_handles.append(plt.Line2D([0], [0], marker=shape, color='w', markerfacecolor=colors[node_type], markersize=10, label=node_type))

    plt.legend(handles=legend_handles)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Assuming you have already created your model and named it 'model'
visualize_model(model, filename='cnn_architecture.png')

"""**Training accuracy vs Validation accuracy & Training loss vs Validation Loss graph**"""

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('Training & Validation Accuracy.png', dpi=300)
plt.show()


# # Plot training and validation loss
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig('Training & Validation Loss.png', dpi=300)
# plt.show()

"""**(Testing accuracy testing loss) vs epoch graph**"""

# Extract testing accuracy and loss from history
test_accuracy = history.history['val_accuracy']
test_loss = history.history['val_loss']

# Create a list of epochs
epochs = range(1, len(test_accuracy) + 1)

# Plot testing accuracy vs epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, test_accuracy, label='Testing Accuracy', color='blue')
plt.title('Testing Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('Testing Accuracy vs No.of epochs.png', dpi=300)
plt.show()

# # Plot testing loss vs epochs
# plt.figure(figsize=(10, 5))
# plt.plot(epochs, test_loss, label='Testing Loss', color='red')
# plt.title('Testing Loss vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig('Testing Loss vs No.of epochs.png', dpi=300)
# plt.show()

# Obtain class names from label encoder
class_names = labelencoder.classes_
class_names

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Obtain class names from label encoder
class_names = labelencoder.classes_

# Step 2: Compute confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Step 3: Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Step 4: Calculate precision, recall, and F1 score
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Step 2: Generate classification report
report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)

# Step 3: Convert classification report to DataFrame
report_df = pd.DataFrame(report).transpose()

# Step 4: Plot classification report as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.title('Classification Report')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Step 5: Save the plot as an image
plt.savefig('classification_report.png', dpi=300)