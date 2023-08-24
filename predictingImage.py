#testing our model on test data and visualizations

import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from PIL import Image
from string import ascii_uppercase
sz=200

# Load the saved model architecture from JSON file
json_file = open('model-bw.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the saved weights into the model
loaded_model.load_weights('model-bw.h5')
print('Model loaded from disk')


# Compile the loaded model (required if you intend to make predictions)
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a function to predict and visualize images
def predict_and_visualize(model, image_paths, target_size, batch_size):
    images = []
    alpha_dict = {}
    for i in range(10):
      alpha_dict[i]=str(i)
    j=10
    for i in ascii_uppercase:
      alpha_dict[j] = i
      j = j + 1
    
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
        images.append(img_array)

    images = np.array(images)
    predictions = model.predict(images, batch_size=batch_size)
    predicted_classes = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 8))
    for i in range(len(image_paths)):
        plt.subplot(4, 5, i + 1)
        img = Image.open(image_paths[i]).convert('L')  # Convert to grayscale
        img = img.resize(target_size)
        plt.imshow(img, cmap='gray')
        plt.title(f'Predicted: {alpha_dict[predicted_classes[i]]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Paths to 20 images from your dataset for prediction
image_paths = [
    
    'newImage.png',
    'newImage1.png',
    'newImage2.png',
    'newImage3.png',
]

# Predict and visualize the batch of images
predict_and_visualize(loaded_model, image_paths, target_size=(sz, sz), batch_size=64)
