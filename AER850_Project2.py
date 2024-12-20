import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU

#Step 1
# Define input image dimensions
img_width, img_height = 500, 500
batch_size = 3

# Defining the paths
train_data = 'C:/Users/hassa/OneDrive/Desktop/School/Fourth Year/AER 850/Project 2/Project 2 Data/Data/train'
validation_data = 'C:/Users/hassa/OneDrive/Desktop/School/Fourth Year/AER 850/Project 2/Project 2 Data/Data/valid'

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,       
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True     
)

# Rescaling validation set
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Generate training data
train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Generate validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)


#Step 2-3
#Initialize the model
model = Sequential()

#Convolutional Layer LeakyReLU
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(500, 500, 3)))
model.add(LeakyReLU(alpha=0.1))  # LeakyReLU activation
model.add(MaxPooling2D(pool_size=(2, 2)))

#Convolutional Layer ELU
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)))
model.add(LeakyReLU(alpha=0.1))  # LeakyReLU activation
model.add(MaxPooling2D(pool_size=(2, 2)))

#Convolutional Layer ReLU
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten layer
model.add(Flatten())

#Fully Connected Dense Layer with ELU and Dropout
model.add(Dense(units=128, activation='elu'))  # ELU activation
model.add(Dropout(0.5))

#Output layer with Softmax for multi-class classification
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Summary of the model
model.summary()

#Step 4

history = model.fit(
    train_generator,
    epochs=10,  
    validation_data=validation_generator
)

# Plot Training and Validation Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1) 
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save('hassan_model.h5')