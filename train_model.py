from data_preprocessing import train_generator, validation_generator
from model_creation import model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# EarlyStopping in caso di overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Training del modello
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Salva i pesi del modello
model.save_weights('model_weights.weights.h5')