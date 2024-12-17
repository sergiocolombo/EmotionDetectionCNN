from data_preprocessing import validation_generator
from model_creation import model

# Load the saved weights into the model
model.load_weights('model_weights.weights.h5')

loss, accuracy = model.evaluate(validation_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')