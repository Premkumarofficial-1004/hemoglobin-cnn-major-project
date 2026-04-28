import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("../my_model.keras")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return "Non-Anemic"
    else:
        return "Anemic"


# Test prediction
result = predict_image("test.jpg")
print("Prediction:", result)