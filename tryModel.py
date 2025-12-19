import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Label Encoding
genus_map = {
    "Agaricus": 0,
    "Amanita": 1,
    "Armillaria": 2,
    "Cortinarius": 3,
    "Lactarius": 4,
    "Marasmius": 5,
    "Suillus": 6,
    "Tricholoma": 7
}

edibility_map = {
    "edible": 0,
    "poisonous": 1
}

# Ekstraksi Fitur menggunakan HSV dan HOG
def extract_features(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    img = cv2.resize(img, img_size)

    # HSV Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None, [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    # HOG
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    return np.hstack([hsv_hist, hog_feat])

# Load Dataset
dataset_path = "Sample Dataset"

X, y_genus, y_edibility = [], [], []

for edibility in os.listdir(dataset_path):
    ed_path = os.path.join(dataset_path, edibility)
    if not os.path.isdir(ed_path):
        continue

    for genus in os.listdir(ed_path):
        genus_path = os.path.join(ed_path, genus)
        if not os.path.isdir(genus_path):
            continue

        for file in os.listdir(genus_path):
            try:
                feat = extract_features(os.path.join(genus_path, file))
                X.append(feat)
                y_genus.append(genus_map[genus])
                y_edibility.append(edibility_map[edibility])
            except:
                continue

X = np.array(X)
y_genus = np.array(y_genus)
y_edibility = np.array(y_edibility)

print("Jumlah data:", X.shape[0])

# Standardisasi Fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, \
y_genus_train, y_genus_test, \
y_ed_train, y_ed_test = train_test_split(
    X_scaled,
    y_genus,
    y_edibility,
    test_size=0.2,
    random_state=42,
    stratify=y_edibility
)

# Modeling
# SVM untuk Klasifikasi Genus
svm_genus = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced'
)

svm_genus.fit(X_train, y_genus_train)

genus_train_pred = svm_genus.predict(X_train)
genus_test_pred = svm_genus.predict(X_test)

# KNN untuk Klasifikasi Edibility dengan Fitur Hierarkis
X_train_ed = np.hstack([
    X_train,
    genus_train_pred.reshape(-1, 1)
])

X_test_ed = np.hstack([
    X_test,
    genus_test_pred.reshape(-1, 1)
])

knn_edibility = KNeighborsClassifier(
    n_neighbors=7,          
    weights='distance',     
    metric='euclidean'
)

knn_edibility.fit(X_train_ed, y_ed_train)

ed_pred = knn_edibility.predict(X_test_ed)

# Evaluasi Model
print("\n=== KLASIFIKASI GENUS (SVM) ===")
print(confusion_matrix(y_genus_test, genus_test_pred))
print(classification_report(y_genus_test, genus_test_pred))

print("\n=== KLASIFIKASI EDIBILITY ===")
print(confusion_matrix(y_ed_test, ed_pred))
print(classification_report(y_ed_test, ed_pred))

#SIMPAN MODEL
os.makedirs("models", exist_ok=True)

joblib.dump(svm_genus, "models/svm_genus.pkl")
joblib.dump(knn_edibility, "models/knn_edibility.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model berhasil disimpan!")