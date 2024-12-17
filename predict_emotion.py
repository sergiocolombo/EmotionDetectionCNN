import numpy as np
from tensorflow.keras.preprocessing import image
from model_creation import model

# Load the saved weights into the model
model.load_weights('model_weights.weights.h5')

# Definisci le classi di emozioni aggiornate
emotion_classes = ['angry', 'happy', 'neutral', 'sad']

def predict_emotion(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_classes[emotion_index]
    return emotion_label

# Esempio di utilizzo
emotion = predict_emotion('test_image.png')
print(f'Predicted emotion: {emotion}')