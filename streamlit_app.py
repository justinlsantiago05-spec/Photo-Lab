# Photo Lab: a small OpenCV photo editor built for Dr. Lee's Intro to Computer Vision class.
# It uses the four ideas from the class 3 notebook:
#   1. Image arithmetic: brightness (cv2.add / cv2.subtract) and contrast (cv2.multiply + np.clip)
#   2. Thresholding: global, inverse, adaptive and Otsu
#   3. Alpha channels and masks: transparent background (threshold mask + bitwise_not + merge)
#   4. Drawing and annotation: caption text (putText) and a frame (rectangle)
#
# Streamlit re-runs this whole file from top to bottom every time a control changes.
# That's what makes the edited picture update live.

import cv2
import numpy as np
import streamlit as st

# Colors are written in BGR order, because that's how OpenCV stores them.
COLORS = {
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Yellow": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Red": (0, 0, 255),
}


def adjust_brightness(img, amount):
    """Add (or subtract) the same number to every pixel. cv2.add stops at 255 on its own."""
    matrix = np.ones(img.shape, dtype="uint8") * abs(int(amount))
    if amount >= 0:
        return cv2.add(img, matrix)
    return cv2.subtract(img, matrix)


def adjust_contrast(img, factor, clip):
    """Multiply every pixel by `factor`. Without clipping, values over 255 wrap around."""
    stretched = cv2.multiply(np.float64(img), np.ones(img.shape, dtype="float64"), scale=factor)
    if clip:
        stretched = np.clip(stretched, 0, 255)
    else:
        stretched = stretched % 256  # show the overflow on purpose, like the Santorini black sky
    return np.uint8(stretched)


def apply_threshold(img, mode, value):
    """Turn the image into pure black and white."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "Global":
        _, out = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
    elif mode == "Global (inverse)":
        _, out = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
    elif mode == "Adaptive":
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 7)
    else:  # Otsu picks the threshold value for you
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)  # back to 3 channels so we can draw in color


def make_transparent(img, cutoff):
    """Make bright background pixels see-through, like the kangaroo sign in class."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)  # white where background is
    alpha = cv2.bitwise_not(bright)                                  # flip: background becomes 0 (invisible)
    b, g, r = cv2.split(img)
    return cv2.merge([b, g, r, alpha])                               # 4th channel = alpha


def annotate(img, caption, color_name, frame):
    """Draw a caption centered at the bottom and an optional frame around the picture."""
    color = COLORS[color_name]
    h, w = img.shape[:2]
    thickness = max(1, w // 300)
    if frame:
        margin = max(5, w // 50)
        cv2.rectangle(img, (margin, margin), (w - margin, h - margin), color, thickness + 1, cv2.LINE_AA)
    if caption.strip():
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = w / 700  # bigger photo, bigger text
        (tw, th), _ = cv2.getTextSize(caption, font, scale, thickness)
        x = max(0, (w - tw) // 2)
        y = h - max(15, h // 20)
        cv2.putText(img, caption, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return img


# ---------------- The web page ----------------
st.set_page_config(page_title="Photo Lab", layout="wide")
st.title("Photo Lab")
st.write("Upload a photo, then use the controls on the left. The edited version updates live.")

# Controls go in the sidebar on the left.
st.sidebar.header("1. Brightness and contrast")
brightness = st.sidebar.slider("Brightness (cv2.add / cv2.subtract)", -100, 100, 0)
contrast = st.sidebar.slider("Contrast (cv2.multiply)", 0.5, 2.0, 1.0, 0.05)
clip = st.sidebar.checkbox("Clip at 255 (turn off to see overflow wrap around)", value=True)

st.sidebar.header("2. Thresholding")
threshold_mode = st.sidebar.selectbox("Threshold type",
                                      ["Off", "Global", "Global (inverse)", "Adaptive", "Otsu (automatic)"])
threshold_value = st.sidebar.slider("Threshold value (Global modes only)", 0, 255, 127)

st.sidebar.header("3. Caption and frame")
caption = st.sidebar.text_input("Caption text", "")
text_color = st.sidebar.selectbox("Text and frame color", list(COLORS))
frame = st.sidebar.checkbox("Draw a frame", value=False)

st.sidebar.header("4. Transparent background")
transparent = st.sidebar.checkbox("Remove bright background", value=False)
bg_cutoff = st.sidebar.slider("Background brightness cutoff", 0, 255, 230)

uploaded = st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a photo to get started.")
else:
    # Turn the uploaded file into an OpenCV image (a grid of numbers in BGR order).
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = original.copy()  # always edit a copy, never the original
    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast, clip)
    if threshold_mode != "Off":
        img = apply_threshold(img, threshold_mode, threshold_value)
    img = annotate(img, caption, text_color, frame)

    if transparent:
        result = make_transparent(img, bg_cutoff)
        shown = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
    else:
        result = img
        shown = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    left, right = st.columns(2)
    left.subheader("Original")
    left.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))  # BGR -> RGB for the browser
    right.subheader("Edited")
    right.image(shown)

    # Let the user download the result as a PNG (PNG keeps the transparent background).
    ok, png = cv2.imencode(".png", result)
    st.download_button("Download edited image (PNG)", png.tobytes(),
                       file_name="photo_lab.png", mime="image/png")
