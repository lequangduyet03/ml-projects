# Machine Learning Projects - Classification & Regression

Dự án Machine Learning với 2 bài toán: Phân loại bệnh tiểu đường và Dự đoán điểm số học sinh.

## 📋 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Kết quả](#kết-quả)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)

## 🎯 Giới thiệu

Repository này chứa 2 dự án Machine Learning:

### 1. Classification - Dự đoán bệnh tiểu đường
- **Thuật toán**: Random Forest Classifier
- **Mục tiêu**: Dự đoán xem bệnh nhân có mắc bệnh tiểu đường hay không
- **Dataset**: diabetes.csv (768 samples, 8 features)
- **Target**: Outcome (0 = Không bị tiểu đường, 1 = Bị tiểu đường)

### 2. Regression - Dự đoán điểm toán học sinh
- **Thuật toán**: Random Forest Regressor
- **Mục tiêu**: Dự đoán điểm toán của học sinh dựa trên các yếu tố khác
- **Dataset**: StudentScore.xls (1000 samples)
- **Target**: Math Score

## 📊 Dataset

### Diabetes Dataset
Các features bao gồm:
- Pregnancies (Số lần mang thai)
- Glucose (Nồng độ đường huyết)
- Blood Pressure (Huyết áp)
- Skin Thickness (Độ dày da)
- Insulin (Nồng độ insulin)
- BMI (Chỉ số khối cơ thể)
- Diabetes Pedigree Function
- Age (Tuổi)

### Student Score Dataset
Các features bao gồm:
- Gender (Giới tính)
- Race/Ethnicity (Chủng tộc)
- Parental Level of Education (Trình độ học vấn của cha mẹ)
- Lunch (Loại bữa trưa)
- Test Preparation Course (Khóa ôn thi)
- Reading Score (Điểm đọc)
- Writing Score (Điểm viết)

## 🛠️ Công nghệ sử dụng

- Python 3.8+
- scikit-learn
- pandas
- numpy

## ⚙️ Cài đặt

1. Clone repository:
```bash
git clone https://github.com/lequangduyet03/ml-projects
cd ml-projects
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 🚀 Cách sử dụng

### Chạy Classification Model
```bash
python classification.py
```

Output:
- Model được lưu tại: `finalized_model.pkl`
- Hiển thị accuracy, precision, recall, F1-score
- Confusion matrix

### Chạy Regression Model
```bash
python regression.py
```

Output:
- Model được lưu tại: `student_score_model.pkl`
- Hiển thị MAE, MSE, R² score
- Dự đoán mẫu với 2 học sinh

### Load model đã lưu
```python
import pickle

# Load classification model
with open('finalized_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)

# Load regression model
with open('student_score_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)
```

## 📈 Kết quả

### Classification Model
- **Best Cross-validation Recall**: ~0.75
- **Test Accuracy**: ~0.77
- **Optimization**: GridSearchCV với 5-fold cross-validation
- **Metric tối ưu**: Recall (để phát hiện nhiều ca bệnh nhất có thể)

### Regression Model
- **MAE**: ~4.0 điểm
- **MSE**: ~25.0
- **R² Score**: ~0.88
- **Optimization**: GridSearchCV với 5-fold cross-validation

## 📁 Cấu trúc thư mục

```
ml-projects/
│
├── classification.py           # Code phân loại bệnh tiểu đường
├── regression.py              # Code dự đoán điểm số
├── requirements.txt           # Thư viện cần thiết
├── README.md                  # File này
│
├── finalized_model.pkl        # Model classification đã train
└── student_score_model.pkl    # Model regression đã train
```

## 🔍 Chi tiết kỹ thuật

### Classification Pipeline
1. Load data
2. Train/Test split (80/20)
3. StandardScaler normalization
4. GridSearchCV với Random Forest
5. Evaluation với nhiều metrics
6. Lưu model

### Regression Pipeline
1. Load data
2. Train/Test split (80/20)
3. Preprocessing Pipeline:
   - Numeric features: Imputation + Scaling
   - Ordinal features: Imputation + Ordinal Encoding
   - Nominal features: Imputation + One-Hot Encoding
4. GridSearchCV với Random Forest
5. Evaluation (MAE, MSE, R²)
6. Test với dữ liệu mẫu
7. Lưu model

## 📝 License

MIT License

## 👤 Author

lequangduyet03 - [GitHub](https://github.com/lequangduyet03)

## 🤝 Contributing

Contributions, issues và feature requests đều được chào đón!

---

⭐ Nếu project này hữu ích, hãy cho 1 star nhé!