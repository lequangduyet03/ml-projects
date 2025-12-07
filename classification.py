"""
Diabetes Classification - Dự đoán bệnh tiểu đường
Sử dụng Random Forest với GridSearchCV để tìm tham số tốt nhất
"""

import pickle
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# 1. LOAD DATA
print("Bước 1: Đọc dữ liệu...")
data = pd.read_csv('diabetes.csv')
print(f"Số dòng: {len(data)}, Số cột: {len(data.columns)}")

# 2. CHIA DỮ LIỆU
print("\nBước 2: Chia train/test...")
x = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(f"Train: {len(x_train)} samples, Test: {len(x_test)} samples")


# 3. CHUẨN HÓA DỮ LIỆU
print("\nBước 3: Chuẩn hóa dữ liệu (scaling)...")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Hoàn tất!")


# 4. TRAIN MODEL VỚI GRIDSEARCHCV
print("\nBước 4: Train model với GridSearchCV...")
print("Đang tìm tham số tốt nhất... (có thể mất vài phút)")

params = {
    "n_estimators": [20, 100, 200],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10],
}

model = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=params,
    scoring="recall",
    cv=5,
    verbose=1,
    n_jobs=-1
)

model.fit(x_train, y_train)

print(f"\nKết quả tốt nhất:")
print(f"  Recall score: {model.best_score_:.4f}")
print(f"  Tham số: {model.best_params_}")


# 5. ĐÁNH GIÁ MODEL
print("\nBước 5: Đánh giá trên test set...")
y_pred = model.predict(x_test)

print(f"Accuracy:  {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {metrics.recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {metrics.f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))


# 6. LƯU MODEL
print("\nBước 6: Lưu model...")
with open('finalized_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Đã lưu model: finalized_model.pkl")

print("\nHoàn tất!")