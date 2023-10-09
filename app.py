import streamlit as st
import io
from PIL import Image
import cv2
import pytesseract
import numpy as np
import torch
from torchvision.models import detection
from collections import Counter

st.set_page_config(
    page_title="Auto NPR",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def instantiate_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="model/last.pt", force_reload=True)
    model.eval()
    model.conf = 0.5
    model.iou = 0.45
    return model

def detect_license_plate(model, image):
    results = model(image, size=640)
    img = np.squeeze(results.render())
    return img

def extract_license_plate_value(image):
    license_plate_value = pytesseract.image_to_string(image, config='--psm 8')
    return license_plate_value.strip()

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

    intersection = x_overlap * y_overlap
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    iou = intersection / (area1 + area2 - intersection + 1e-6)  # Adding a small constant to avoid division by zero
    return iou

main_image = Image.open('static/main_banner.png')

st.image(main_image, use_column_width='auto')
st.title('Automatic Number Plate Recognition')

st.info('The Live Feed from Web-Camera will take some time to load')
live_feed = st.checkbox('Start Web-Camera')
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

license_plate_display = st.empty()
license_plate_input = st.empty()

if live_feed:
    detected_license_plates = []
    license_plate_value = None

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # License plate detection
            model = instantiate_model()  # Instantiate the model
            img_detected = detect_license_plate(model, img)
            img_BGR = cv2.cvtColor(img_detected, cv2.COLOR_RGB2BGR)

            # License plate value detection and post-processing
            results = model(img, size=640)  # Use the same model for detection
            results_np = np.array(results.pred[0])

            if len(results_np) > 0:
                filtered_boxes = []
                for det in results_np:
                    x_min, y_min, x_max, y_max, conf, cls = det
                    if conf > 0.5 and int(cls) == 0:
                        box = [x_min, y_min, x_max, y_max]
                        overlap = False
                        for filtered_box in filtered_boxes:
                            iou = calculate_iou(box, filtered_box)
                            if iou > 0.5:
                                overlap = True
                                break
                        if not overlap:
                            filtered_boxes.append(box)
                            license_plate_region = img_BGR[int(y_min):int(y_max), int(x_min):int(x_max)]
                            license_plate_value = extract_license_plate_value(license_plate_region)
                            detected_license_plates.append(license_plate_value)
                            break

            for value in detected_license_plates:
                if license_plate_value is not None and value == license_plate_value:
                    editable_license_plate_value = license_plate_input.text_input("Edit License Plate Value:", value)
                    if editable_license_plate_value != value:
                        license_plate_value = editable_license_plate_value

            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            FRAME_WINDOW.image(frame)

        else:
            break

else:
    cap.release()
    cv2.destroyAllWindows()
    st.warning('The Web-Camera is currently disabled')

st.markdown("<br><hr><center>Automatic Number Plate Recognition</center><hr>", unsafe_allow_html=True)
