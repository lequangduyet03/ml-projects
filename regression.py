"""
Student Score Regression - Dự đoán điểm toán học sinh
Sử dụng Random Forest với Pipeline và GridSearchCV
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


# 1. LOAD DATA
print("Bước 1: Đọc dữ liệu...")
data = pd.read_csv('StudentScore.xls')
print(f"Số dòng: {len(data)}, Số cột: {len(data.columns)}")
print(f"Các cột: {list(data.columns)}")


# 2. CHIA DỮ LIỆU
print("\nBước 2: Chia train/test...")
x = data.drop('math score', axis=1)  # Features
y = data['math score']  # Target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(f"Train: {len(x_train)} samples, Test: {len(x_test)} samples")


# 3. TẠO PREPROCESSING PIPELINE
print("\nBước 3: Tạo preprocessing pipeline...")

# Pipeline cho số (numeric features)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline cho categorical có thứ tự (ordinal)
education_order = [
    "some high school", 
    "high school",
    "some college",
    "associate's degree", 
    "bachelor's degree", 
    "master's degree"
]

ord_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=[education_order, ['female', 'male'], 
                   ['free/reduced', 'standard'], ['none', 'completed']],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

# Pipeline cho categorical không có thứ tự (nominal)
nom_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Kết hợp tất cả
preprocessor = ColumnTransformer([
    ('num', num_pipeline, ['reading score', 'writing score']),
    ('ord', ord_pipeline, ['parental level of education', 'gender', 'lunch', 'test preparation course']),
    ('nom', nom_pipeline, ['race/ethnicity'])
])

print("Hoàn tất pipeline!")


# 4. TRAIN MODEL VỚI GRIDSEARCHCV
print("\nBước 4: Train model với GridSearchCV...")
print("Đang tìm tham số tốt nhất... (có thể mất vài phút)")

# Tạo full pipeline (preprocessing + model)
full_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Các tham số cần tìm
params = {
    'preprocessing__num__imputer__strategy': ['median', 'mean'],
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20],
}

model = GridSearchCV(
    full_pipeline,
    param_grid=params,
    scoring='neg_mean_absolute_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

model.fit(x_train, y_train)

print(f"\nKết quả tốt nhất:")
print(f"  MAE score: {-model.best_score_:.4f}")
print(f"  Tham số: {model.best_params_}")


# 5. ĐÁNH GIÁ MODEL
print("\nBước 5: Đánh giá trên test set...")
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"MSE (Mean Squared Error):  {mse:.4f}")
print(f"R² Score:                  {r2:.4f}")


# 6. TEST VỚI DỮ LIỆU MẪU
print("\nBước 6: Test với dữ liệu mẫu...")

sample_data = pd.DataFrame({
    'gender': ['male', 'female'],
    'race/ethnicity': ['group A', 'group E'],
    'parental level of education': ['high school', "bachelor's degree"],
    'lunch': ['standard', 'standard'],
    'test preparation course': ['none', 'completed'],
    'reading score': [80, 85],
    'writing score': [90, 92]
})

predictions = model.predict(sample_data)
print("Dự đoán điểm toán:")
for i, pred in enumerate(predictions):
    print(f"  Học sinh {i+1}: {pred:.2f} điểm")


# 7. LƯU MODEL
print("\nBước 7: Lưu model...")
import pickle

with open('student_score_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Đã lưu model: student_score_model.pkl")

print("\nHoàn tất!")