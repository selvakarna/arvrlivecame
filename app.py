import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import tempfile

st.set_page_config(layout="wide")
st.title("🎭 AI Live Camera - AR Filters + Background Remove")

# Load Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---- UI Controls ----
filter_option = st.selectbox(
    "Select Filter",
    ["None", "Glasses", "Funny Nose", "Cartoon Effect"]
)

remove_bg = st.checkbox("Remove Background (AI)")

size_scale = st.slider("Filter Size", 50, 400, 200)
x_adjust = st.slider("Move Left/Right", -200, 200, 0)
y_adjust = st.slider("Move Up/Down", -200, 200, 0)

picture = st.camera_input("📸 Take a picture")

# ---- Process Image ----
if picture:
    image = Image.open(picture).convert("RGBA")

    # Background removal
    if remove_bg:
        image = remove(image)

    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    result = img_array.copy()

    # ---- Cartoon Effect ----
    if filter_option == "Cartoon Effect":
        gray_img = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray_img, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(img_bgr, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        result = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGBA)

    # ---- Glasses / Funny Filter ----
    if filter_option in ["Glasses", "Funny Nose"]:
        for (x, y, w, h) in faces:

            overlay = np.zeros((size_scale, size_scale, 4), dtype=np.uint8)

            if filter_option == "Glasses":
                cv2.rectangle(overlay, (0, 40), (size_scale, 120),
                              (0, 0, 0, 255), -1)

            if filter_option == "Funny Nose":
                cv2.circle(overlay,
                           (size_scale//2, size_scale//2),
                           size_scale//4,
                           (0, 0, 255, 255), -1)

            overlay_img = Image.fromarray(overlay, "RGBA")
            result_pil = Image.fromarray(result).convert("RGBA")

            x_pos = x + w//2 - size_scale//2 + x_adjust
            y_pos = y + h//3 + y_adjust

            result_pil.paste(overlay_img, (x_pos, y_pos), overlay_img)
            result = np.array(result_pil)

    st.image(result, caption="🎉 Final Output", use_column_width=True)

    # Download
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(result).save(temp_file.name)

    st.download_button(
        "📥 Download Image",
        data=open(temp_file.name, "rb"),
        file_name="final_output.png"
    )
