import numpy as np
import shutil
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications import VGG16, MobileNet, DenseNet121
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to temporarily store images
IMAGE_STORAGE_PATH = "temp_images/"
os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)

def align_and_extract_circuit(image_path):
    """
    Aligns the image using a perspective transform and extracts the blue circuit.
    """
    image = cv2.imread(image_path)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    phone_contour = contours[0]
    rect = cv2.minAreaRect(phone_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    ordered_box = order_points(box)
    (tl, tr, br, bl) = ordered_box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    max_width = int(max(widthA, widthB))
    max_height = int(max(heightA, heightB))
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_box, dst_pts)
    aligned = cv2.warpPerspective(orig, M, (max_width, max_height))
    hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        circuit_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(circuit_contour)
        circuit = aligned[y:y+h, x:x+w]
        cropped_path = image_path.replace(".jpg", "_cropped.jpg")
        cv2.imwrite(cropped_path, circuit)
        return cropped_path
    return None

class ImageSimilarity:
    vgg_model = VGG16(weights='imagenet', include_top=False)
    mobilenet_model = MobileNet(weights='imagenet', include_top=False)
    densenet_model = DenseNet121(weights='imagenet', include_top=False)

    @staticmethod
    def extract_features(image_path, model, preprocess_function):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_function(img_array)
        features = model.predict(img_array)
        return features.flatten()

    @staticmethod
    def calculate_similarity(image_path1, image_path2, model):
        selected_model = [ImageSimilarity.mobilenet_model, ImageSimilarity.vgg_model, ImageSimilarity.densenet_model][model]
        preprocess_function = [mobilenet_preprocess_input, vgg_preprocess_input, densenet_preprocess_input][model]
        feature1 = ImageSimilarity.extract_features(image_path1, selected_model, preprocess_function)
        feature2 = ImageSimilarity.extract_features(image_path2, selected_model, preprocess_function)
        similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        return float(similarity)

@app.post("/check_similarity/")
async def check_similarity(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    model: int = Form(...)
):
    path1 = os.path.join(IMAGE_STORAGE_PATH, image1.filename)
    path2 = os.path.join(IMAGE_STORAGE_PATH, image2.filename)
    with open(path1, "wb") as f:
        shutil.copyfileobj(image1.file, f)
    with open(path2, "wb") as f:
        shutil.copyfileobj(image2.file, f)

    similarity = ImageSimilarity.calculate_similarity(path1, path2, model)
    percentage = round(similarity * 100, 2)

    # Optionally remove the files after processing
    os.remove(path1)
    os.remove(path2)

    return JSONResponse({"similarity_percentage": percentage})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)