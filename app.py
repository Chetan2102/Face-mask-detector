import streamlit as st
import cv2
import numpy as np

from predict import predict

st.title("Face Mask Detection")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    bytes_data = uploaded_file.read()

    image = cv2.imdecode(
        np.frombuffer(bytes_data,np.uint8),
        cv2.IMREAD_COLOR
    )

    label,confidence = predict(image)

    st.image(
        cv2.cvtColor(image,cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    st.subheader(label)

    st.write(f"Confidence : {confidence*100:.2f}%")