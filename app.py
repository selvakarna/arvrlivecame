import streamlit as st
from PIL import Image
import numpy as np
import cv2
from rembg import remove
import tempfile

st.title("AI Live Camera - AR Virtual Overlay App")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Capture image
picture = st.camera_input("Take a picture")

# Upload virtual object (PNG transparent recommended)
virtual_file = st.file_uploader("Upload Virtual Object (PNG)", type=["png"])

# Controls
size_scale = st.slider("Object Size", 50, 400, 200)
x_adjust = st.slider("Move Left/Right", -300, 300, 0)
y_adjust = st.slider("Move Up/Down", -300, 300, 0)

remove_bg = st.checkbox("Remove Background (AI)")

if picture:
    image = Image.open(picture).convert("RGBA")

    if remove_bg:
        image = remove(image)

    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    result_img = image.copy()

    if virtual_file and len(faces) > 0:
        virtual_img = Image.open(virtual_file).convert("RGBA")
        virtual_img = virtual_img.resize((size_scale, size_scale))

        for (x, y, w, h) in faces:
            x_pos = x + int(w / 2) - int(size_scale / 2) + x_adjust
            y_pos = y + int(h / 3) + y_adjust

            result_img.paste(virtual_img, (x_pos, y_pos), virtual_img)

    st.image(result_img, caption="Final Output")

    # Save and download
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    result_img.save(temp_file.name)

    st.download_button(
        "Download Image",
        data=open(temp_file.name, "rb"),
        file_name="ar_output.png",
    )
