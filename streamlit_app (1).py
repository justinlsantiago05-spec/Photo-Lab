# Photo Lab: an OpenCV photo editor built for Dr. Lee's Intro to Computer Vision class.
#
# Concepts from class that this app uses:
#   1. Image arithmetic: brightness (cv2.add / cv2.subtract) and contrast (cv2.multiply + np.clip)
#   2. Thresholding: global, inverse, adaptive and Otsu
#   3. Alpha channels and masks: transparent backgrounds (threshold or HSV color mask,
#      bitwise_not, merge as a 4th channel)
#   4. Drawing and annotation: caption (putText), box and frame (rectangle)
#   5. Histograms (Course Labs CV 03): brightness histogram of the original vs the edit
#   6. Color segmentation by hue (Course Labs CV 03): remove one background color in HSV
#   7. Contours: keep only the main subject when removing a background
#   8. Bonus: YOLO object detection (YOLOv4-tiny, the small version of the model from class)
#
# Streamlit re-runs this whole file from top to bottom every time a control changes.
# That's what makes the edited picture update live.

import os
import tempfile
import urllib.request

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================================
# Settings
# ======================================================================

# Colors are written in BGR order, because that's how OpenCV stores them.
COLORS = {
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Yellow": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Red": (0, 0, 255),
}

# Sample photos (most come from Dr. Lee's class image folder on GitHub).
SAMPLES = {
    "Santorini (try contrast)": "https://raw.githubusercontent.com/fenago/cv2/main/Mod2/Santorini.jpg",
    "Sheet music in shadow (try Clean up a document)": "https://raw.githubusercontent.com/fenago/cv2/main/Mod2/Sheet_Music_Test-1.jpg",
    "Kangaroo sign (try Remove background)": "https://raw.githubusercontent.com/fenago/cv2/main/Mod2/kangaroo.jpg",
    "Horses (try YOLO object detection)": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg",
}

# YOLOv4-tiny: a small (about 24 MB) version of the YOLO model Dr. Lee ran in class.
YOLO_FILES = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}

MAX_SIDE = 1200  # big photos are shrunk to this size so the app stays fast

# Every control's starting value. Presets and the Reset button change these.
DEFAULTS = {
    "brightness": 0,
    "contrast": 1.0,
    "clip": True,
    "show_overflow": False,
    "thr_mode": "Off",
    "thr_value": 127,
    "caption": "",
    "caption_pos": "Bottom",
    "ink": "White",
    "frame": False,
    "show_box": False,
    "box_x": 25,
    "box_y": 20,
    "box_w": 50,
    "box_h": 60,
    "bg_mode": "Off",
    "bg_cutoff": 230,
    "bg_source": "Auto (use the corner color)",
    "bg_color": "#f0b37e",
    "bg_tol": 12,
    "keep_main": True,
    "yolo": False,
    "yolo_conf": 0.4,
}

# ======================================================================
# Image functions
# ======================================================================


def shrink(img):
    """Shrink very large photos (keeping their shape) so every slider stays fast."""
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def adjust_brightness(img, amount):
    """Add (or subtract) the same number to every pixel. cv2.add stops at 255 on its own.
    Also returns which pixels hit the 0 or 255 limit."""
    matrix = np.ones(img.shape, dtype="uint8") * abs(int(amount))
    raw = img.astype(np.int32) + int(amount)
    overflow = ((raw > 255) | (raw < 0)).any(axis=2)
    if amount >= 0:
        return cv2.add(img, matrix), overflow
    return cv2.subtract(img, matrix), overflow


def adjust_contrast(img, factor, clip):
    """Multiply every pixel by `factor`. Without clipping, values over 255 wrap around.
    Also returns which pixels went past 255."""
    stretched = cv2.multiply(np.float64(img), np.ones(img.shape, dtype="float64"), scale=factor)
    overflow = (stretched > 255).any(axis=2)
    if clip:
        stretched = np.clip(stretched, 0, 255)
    else:
        stretched = stretched % 256  # show the overflow on purpose, like the Santorini black sky
    return np.uint8(stretched), overflow


def apply_threshold(img, mode, value):
    """Turn the image into pure black and white."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "Global":
        _, out = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
    elif mode == "Global (inverse)":
        _, out = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
    elif mode == "Adaptive":
        # block size grows with the photo, so it works on small and large images
        block = max(11, (min(gray.shape) // 40) | 1)
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, 7)
    else:  # Otsu picks the threshold value for you
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)  # back to 3 channels so we can draw in color


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def bright_mask(img, cutoff):
    """Background = every pixel brighter than the cutoff (the kangaroo method from class)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)
    return mask


def color_mask(img, bgr, tolerance):
    """Background = every pixel whose hue is close to one color (color segmentation in HSV)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    s_low, v_low = max(s - 80, 0), max(v - 80, 0)
    low, high = h - tolerance, h + tolerance
    # OpenCV hue goes 0-179 and wraps around (red sits at both ends), so split the range if needed.
    ranges = [(max(low, 0), min(high, 179))]
    if low < 0:
        ranges.append((180 + low, 179))
    if high > 179:
        ranges.append((0, high - 180))
    mask = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array([lo, s_low, v_low], np.uint8), np.array([hi, 255, 255], np.uint8))
    return mask


def keep_main_subject(foreground):
    """Keep only the biggest shape (contour) and fill its holes. Drops specks of leftover background."""
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return foreground
    biggest = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(foreground)
    cv2.drawContours(cleaned, [biggest], -1, 255, thickness=cv2.FILLED)
    return cleaned


def place_text(img, text, position, color):
    """Draw a caption centered left-to-right, at the top, middle or bottom."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, w // 300)
    scale = max(0.5, w / 700)  # bigger photo, bigger text (but never too tiny to read)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw > 0.9 * w:  # caption too wide? shrink it so it fits
        scale = scale * 0.9 * w / tw
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (w - tw) // 2)
    pad = max(15, h // 20)
    y = {"Top": th + pad, "Middle": (h + th) // 2, "Bottom": h - pad}[position]
    # A dark (or light) outline first, then the colored letters on top, so the caption is readable anywhere.
    outline = (0, 0, 0) if sum(color) > 300 else (255, 255, 255)
    cv2.putText(img, text, (x, y), font, scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


@st.cache_resource(show_spinner="Downloading the YOLO model (about 24 MB, first time only)...")
def load_yolo():
    folder = os.path.join(tempfile.gettempdir(), "photo_lab_yolo")
    os.makedirs(folder, exist_ok=True)
    for name, url in YOLO_FILES.items():
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
    net = cv2.dnn.readNetFromDarknet(os.path.join(folder, "yolov4-tiny.cfg"),
                                     os.path.join(folder, "yolov4-tiny.weights"))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    classes = open(os.path.join(folder, "coco.names")).read().strip().split("\n")
    return net, classes


def detect_objects(img, min_confidence):
    """Run YOLO, then keep boxes above the confidence level (same steps as the class notebook)."""
    net, classes = load_yolo()
    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = np.vstack(net.forward(net.getUnconnectedOutLayersNames()))
    boxes, confidences, class_ids = [], [], []
    for det in outputs:
        scores = det[5:]
        cid = int(np.argmax(scores))
        conf = float(scores[cid])
        if conf > min_confidence:
            cx, cy, w, h = det[:4] * np.array([W, H, W, H])
            boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
            confidences.append(conf)
            class_ids.append(cid)
    keep = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)  # remove duplicate boxes
    return [(boxes[i], classes[class_ids[i]], confidences[i]) for i in np.array(keep).flatten()]


def draw_detections(img, detections):
    thickness = max(2, img.shape[1] // 300)
    scale = max(0.5, img.shape[1] / 1200)
    for (x, y, w, h), label, conf in detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), thickness)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        top = max(y, th + 6)
        cv2.rectangle(img, (x, top - th - 6), (x + tw + 4, top), (0, 255, 255), cv2.FILLED)
        cv2.putText(img, text, (x + 2, top - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def checkerboard_under(rgba):
    """Put a gray-and-white checkerboard behind see-through pixels, so transparency is visible."""
    h, w = rgba.shape[:2]
    square = max(8, min(h, w) // 25)
    yy, xx = np.indices((h, w)) // square
    board = np.where(((yy + xx) % 2 == 0)[..., None], 255, 200).astype(np.float64)
    alpha = rgba[:, :, 3:4] / 255.0
    return np.uint8(rgba[:, :, :3] * alpha + board * (1 - alpha))


def gray_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()


@st.cache_data(show_spinner="Loading sample photo...")
def load_sample(url):
    with urllib.request.urlopen(url) as response:
        data = response.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ======================================================================
# Presets and Reset (they change the controls' saved values)
# ======================================================================


def reset_all():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def preset(**changes):
    reset_all()
    for key, value in changes.items():
        st.session_state[key] = value


for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# ======================================================================
# The web page
# ======================================================================

st.set_page_config(page_title="Photo Lab", page_icon="📷", layout="wide")
st.title("📷 Photo Lab")
st.caption("An OpenCV photo editor. Upload a photo (or pick a sample), then use the controls on the left. "
           "The edited version updates live.")

# ---------------- Sidebar controls ----------------
st.sidebar.subheader("Quick presets")
st.sidebar.button("📄 Clean up a document", on_click=preset, kwargs={"thr_mode": "Adaptive"}, width="stretch")
st.sidebar.button("🎨 High-contrast poster", on_click=preset, kwargs={"contrast": 1.5, "brightness": 10}, width="stretch")
st.sidebar.button("✂️ Remove background", on_click=preset, kwargs={"bg_mode": "One color (HSV)"}, width="stretch")
st.sidebar.button("↺ Reset everything", on_click=reset_all, width="stretch")

with st.sidebar.expander("☀️ Brightness and contrast", expanded=True):
    brightness = st.slider("Brightness (cv2.add / cv2.subtract)", -100, 100, key="brightness")
    contrast = st.slider("Contrast (cv2.multiply)", 0.5, 2.0, step=0.05, key="contrast")
    clip = st.checkbox("Clip at 255 (turn off to see overflow wrap around)", key="clip")
    show_overflow = st.checkbox("Highlight pixels pushed past the 0–255 limit in red", key="show_overflow")

with st.sidebar.expander("🌗 Thresholding (black and white)", expanded=True):
    thr_mode = st.selectbox("Threshold type", ["Off", "Global", "Global (inverse)", "Adaptive", "Otsu (automatic)"],
                            key="thr_mode")
    if thr_mode in ("Global", "Global (inverse)"):
        thr_value = st.slider("Threshold value", 0, 255, key="thr_value")
    else:
        thr_value = st.session_state["thr_value"]

with st.sidebar.expander("✏️ Caption, box and frame", expanded=False):
    caption = st.text_input("Caption text", key="caption")
    caption_pos = st.radio("Caption position", ["Top", "Middle", "Bottom"], horizontal=True, key="caption_pos")
    ink = st.selectbox("Color", list(COLORS), key="ink")
    frame = st.checkbox("Draw a frame around the photo", key="frame")
    show_box = st.checkbox("Draw a box (to highlight something)", key="show_box")
    if show_box:
        box_x = st.slider("Box left edge (% across)", 0, 100, key="box_x")
        box_y = st.slider("Box top edge (% down)", 0, 100, key="box_y")
        box_w = st.slider("Box width (%)", 1, 100, key="box_w")
        box_h = st.slider("Box height (%)", 1, 100, key="box_h")

with st.sidebar.expander("✂️ Transparent background", expanded=st.session_state["bg_mode"] != "Off"):
    bg_mode = st.selectbox("Remove", ["Off", "Bright pixels (brightness threshold)", "One color (HSV)"], key="bg_mode")
    if bg_mode == "Bright pixels (brightness threshold)":
        bg_cutoff = st.slider("Brightness cutoff", 0, 255, key="bg_cutoff")
    elif bg_mode == "One color (HSV)":
        bg_source = st.radio("Which color?", ["Auto (use the corner color)", "Pick a color"], key="bg_source")
        if bg_source == "Pick a color":
            bg_color = st.color_picker("Background color", key="bg_color")
        bg_tol = st.slider("Hue tolerance (how close counts as the same color)", 1, 40, key="bg_tol")
    if bg_mode != "Off":
        keep_main = st.checkbox("Keep only the main subject (fill holes, drop specks)", key="keep_main")

with st.sidebar.expander("🔍 Bonus: YOLO object detection", expanded=False):
    use_yolo = st.checkbox("Find objects with YOLO", key="yolo")
    if use_yolo:
        yolo_conf = st.slider("Minimum confidence", 0.1, 0.9, step=0.05, key="yolo_conf")
        st.caption("Uses YOLOv4-tiny, a small, fast version of the YOLO model from class. "
                   "It recognizes 80 kinds of everyday objects.")

# ---------------- Choose the photo ----------------
up_col, sample_col = st.columns([3, 2])
uploaded = up_col.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"])
sample_name = sample_col.selectbox("…or try a sample photo", ["(none)"] + list(SAMPLES))

original = None
if uploaded is not None:
    original = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
elif sample_name != "(none)":
    try:
        original = load_sample(SAMPLES[sample_name])
    except Exception:
        st.error("Couldn't download that sample photo. Try another one, or upload your own.")

if original is None:
    st.info("⬆️ Upload a photo, or pick a sample photo, to get started.")
    st.stop()

original = shrink(original)
notes = []  # plain-English explanations of what each active effect is doing

# ---------------- The editing pipeline ----------------
img = original.copy()  # always edit a copy, never the original

img, over_b = adjust_brightness(img, brightness)
img, over_c = adjust_contrast(img, contrast, clip)
overflow = over_b | over_c
if brightness != 0:
    notes.append(f"**Brightness:** `cv2.{'add' if brightness > 0 else 'subtract'}` changed every pixel by "
                 f"{abs(brightness)}. OpenCV stops at 0 and 255 on its own, so nothing wraps.")
if contrast != 1.0:
    if clip:
        notes.append(f"**Contrast:** every pixel was multiplied by {contrast:.2f}, then `np.clip` capped anything "
                     "over 255 at pure white.")
    else:
        notes.append(f"**Contrast without clipping:** every pixel was multiplied by {contrast:.2f}. Pixels that went "
                     "past 255 **wrapped around** to dark values (like 276 → 20), which causes the strange dark or "
                     "colored patches in bright areas.")
if show_overflow:
    pct = 100 * overflow.mean()
    notes.append(f"**Red pixels:** {pct:.1f}% of the photo was pushed past the 0–255 limit.")

color_version = img.copy()  # keep a color copy for the background remover

if thr_mode != "Off":
    img = apply_threshold(img, thr_mode, thr_value)
    notes.append({
        "Global": f"**Global threshold:** every pixel brighter than {thr_value} became white, everything else black. "
                  "One cutoff for the whole photo, so shadows can fool it.",
        "Global (inverse)": f"**Inverse threshold:** pixels darker than {thr_value} became white (dark ink stands "
                            "out), everything else black.",
        "Adaptive": "**Adaptive threshold:** each pixel is compared to its own neighborhood, so a shadow on one part "
                    "of the photo doesn't fool it. That's why it cleans up the sheet music.",
        "Otsu (automatic)": "**Otsu threshold:** OpenCV looked at the histogram and picked the cutoff that best "
                            "splits dark pixels from bright ones.",
    }[thr_mode])

if show_overflow:
    img[overflow] = (0, 0, 255)

# Background mask (worked out before anything is drawn on the photo)
alpha = None
if bg_mode != "Off":
    if bg_mode == "Bright pixels (brightness threshold)":
        background = bright_mask(color_version, bg_cutoff)
        notes.append(f"**Background removal (brightness):** pixels brighter than {bg_cutoff} became see-through. "
                     "Anything bright counts, even parts of the subject (like an elephant's tusks).")
    else:
        if st.session_state["bg_source"] == "Auto (use the corner color)":
            corner = color_version[: max(5, color_version.shape[0] // 30), : max(5, color_version.shape[1] // 30)]
            target = tuple(int(c) for c in np.median(corner.reshape(-1, 3), axis=0))
        else:
            target = hex_to_bgr(st.session_state["bg_color"])
        background = color_mask(color_version, target, st.session_state["bg_tol"])
        notes.append("**Background removal (color):** the photo was converted to HSV, and pixels whose **hue** is "
                     "close to the background color became see-through. Unlike the brightness method, a bright "
                     "part of the subject in a different color survives.")
    foreground = cv2.bitwise_not(background)  # flip: background 0 (invisible), subject 255 (visible)
    if st.session_state["keep_main"]:
        foreground = keep_main_subject(foreground)
        notes.append("**Keep main subject:** `cv2.findContours` found the outlines of the visible shapes, and only "
                     "the biggest one was kept, with its holes filled.")
    alpha = foreground

# Drawing and annotation
before_drawing = img.copy()
if use_yolo:
    try:
        detections = detect_objects(color_version, yolo_conf)
        draw_detections(img, detections)
        found = ", ".join(sorted({label for _, label, _ in detections})) or "nothing above that confidence"
        notes.append(f"**YOLO** found {len(detections)} object(s): {found}. Each box comes with the model's "
                     "confidence (1.00 = completely sure).")
    except Exception as error:
        st.warning(f"YOLO couldn't run here: {error}")
h, w = img.shape[:2]
color = COLORS[ink]
line = max(2, w // 300)
if frame:
    margin = max(5, w // 50)
    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), color, line, cv2.LINE_AA)
if show_box:
    x1, y1 = int(w * box_x / 100), int(h * box_y / 100)
    x2, y2 = min(w - 1, x1 + int(w * box_w / 100)), min(h - 1, y1 + int(h * box_h / 100))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line, cv2.LINE_AA)
if caption.strip():
    place_text(img, caption, caption_pos, color)
if frame or show_box or caption.strip():
    notes.append("**Drawing:** `cv2.rectangle` and `cv2.putText` changed the pixels along the shapes and letters "
                 "to the chosen color. Positions are (x, y): across first, then down from the top.")

# Put it together
if alpha is not None:
    drawn = (img != before_drawing).any(axis=2)
    alpha[drawn] = 255  # keep anything we drew visible, even on the see-through background
    b, g, r = cv2.split(img)
    result = cv2.merge([b, g, r, alpha])  # 4th channel = alpha
    shown = checkerboard_under(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
else:
    result = img
    shown = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# ---------------- Show the results ----------------
left, right = st.columns(2)
left.subheader("Original")
left.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), width="stretch")  # BGR -> RGB for the browser
right.subheader("Edited")
right.image(shown, width="stretch")
if alpha is not None:
    right.caption("The checkerboard shows which parts are see-through. The downloaded PNG keeps them transparent.")

ok, png = cv2.imencode(".png", result)
right.download_button("⬇️ Download edited image (PNG)", png.tobytes(), file_name="photo_lab.png", mime="image/png")

st.subheader("What's happening")
if notes:
    for note in notes:
        st.markdown(f"- {note}")
else:
    st.markdown("- Nothing yet. Move a slider, pick a preset, or turn on a tool in the sidebar.")

st.subheader("Brightness histogram")
st.caption("How many pixels sit at each brightness level, from 0 (black) on the left to 255 (white) on the right. "
           "Raising contrast stretches the shape outward; thresholding squeezes everything into two spikes at "
           "0 and 255.")
hist = pd.DataFrame({"Original": gray_histogram(original), "Edited": gray_histogram(img)})
hist.index.name = "Brightness (0 = black, 255 = white)"
st.line_chart(hist, color=["#3987e5", "#d95926"], x_label="Brightness (0 = black, 255 = white)",
              y_label="Number of pixels", height=280)
