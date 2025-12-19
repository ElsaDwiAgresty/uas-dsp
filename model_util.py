import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load model
svm_genus = joblib.load("models/svm_genus.pkl")
knn_edibility = joblib.load("models/knn_edibility.pkl")
scaler = joblib.load("models/scaler.pkl")

GENUS_LABEL = {
    0: "Agaricus",
    1: "Amanita",
    2: "Armillaria",
    3: "Cortinarius",
    4: "Lactarius",
    5: "Marasmius",
    6: "Suillus",
    7: "Tricholoma"
}

EDIBILITY_LABEL = {
    0: "Edible",
    1: "Poisonous"
}

def extract_features(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None, [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    return np.hstack([hsv_hist, hog_feat])

def predict_image(image_path):
    feat = extract_features(image_path)
    feat_scaled = scaler.transform([feat])

    genus_pred = svm_genus.predict(feat_scaled)[0]

    # Hierarchical feature
    feat_knn = np.hstack([feat_scaled, [[genus_pred]]])

    edibility_pred = knn_edibility.predict(feat_knn)[0]

    return {
        "genus": GENUS_LABEL[genus_pred],
        "edibility": EDIBILITY_LABEL[edibility_pred]
    }