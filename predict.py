import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("face_mask_detector.keras")


def predict(image):

    image = cv2.resize(image,(128,128))
    image = image/255.0

    image = image.reshape(1,128,128,3)

    prediction = model.predict(image,verbose=0)

    confidence = np.max(prediction)

    label = np.argmax(prediction)

    if label==1:
        return "Mask",confidence

    return "No Mask",confidence