# Machine Learning Projects - Classification & Regression

Dự án Machine Learning với 3 bài toán: Phân loại bệnh tiểu đường, Dự đoán điểm số học sinh, và Phân loại cấp độ nghề nghiệp.

## 📋 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Kết quả](#kết-quả)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)

## 🎯 Giới thiệu

Repository này chứa 3 dự án Machine Learning:

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

### 3. Job Classification - Phân loại cấp độ nghề nghiệp
- **Thuật toán**: Random Forest Classifier với Feature Selection
- **Mục tiêu**: Dự đoán cấp độ nghề nghiệp (career level) từ thông tin công việc
- **Dataset**: final_project.ods
- **Target**: career_level (6 classes)
- **Kỹ thuật đặc biệt**: 
  - TF-IDF cho text features (title, description)
  - One-Hot Encoding cho categorical features
  - Random Over-sampling để xử lý imbalanced data
  - Chi-square feature selection

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

### Job Classification Dataset
Các features bao gồm:
- **title**: Chức danh công việc (text)
- **description**: Mô tả công việc (text - unigrams + bigrams)
- **location**: Vị trí địa lý (categorical)
- **function**: Chức năng/phòng ban (categorical)
- **industry**: Ngành nghề (categorical)

Target classes (career_level):
- bereichsleiter
- director_business_unit_leader
- manager_team_leader
- managing_director_small_medium_company
- senior_specialist_or_project_manager
- specialist

## 🛠️ Công nghệ sử dụng

- Python 3.8+
- scikit-learn
- pandas
- numpy
- imbalanced-learn (imblearn)
- openpyxl (đọc file .ods)

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

### Chạy Job Classifier Model
```bash
python job_classifier.py
```

Output:
- Hiển thị phân phối class trước và sau over-sampling
- Classification report với precision, recall, F1-score cho từng class
- Overall accuracy: ~76%

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

### Job Classifier Model
- **Overall Accuracy**: ~76%
- **Best performing class**: senior_specialist_or_project_manager (F1=0.87)
- **Challenges**: Imbalanced data với một số class rất ít samples
- **Techniques used**:
  - Random Over-sampling để cân bằng training data
  - TF-IDF với unigrams + bigrams cho text processing
  - Chi-square feature selection (top 5% features)
  - Random Forest với 100 trees

**Performance by class:**
- senior_specialist_or_project_manager: F1=0.87 ✅
- manager_team_leader: F1=0.69 ✅
- bereichsleiter: F1=0.19 ⚠️
- director_business_unit_leader: F1=0.25 ⚠️
- specialist: F1=0.00 ❌

## 📁 Cấu trúc thư mục

```
ml-projects/
│
├── classification.py           # Code phân loại bệnh tiểu đường
├── regression.py              # Code dự đoán điểm số
├── job_classifier.py          # Code phân loại cấp độ nghề nghiệp
├── requirements.txt           # Thư viện cần thiết
├── README.md                  # File này
│
├── finalized_model.pkl        # Model classification đã train
├── student_score_model.pkl    # Model regression đã train
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

### Job Classifier Pipeline
1. Load data và xử lý missing values
2. Location preprocessing (extract state code)
3. Train/Test split (80/20, stratified)
4. Random Over-sampling (cân bằng classes trong training set)
5. Feature Engineering:
   - TF-IDF vectorization cho title
   - TF-IDF với unigrams+bigrams cho description (min_df=0.01, max_df=0.99)
   - One-Hot Encoding cho location, function, industry
6. Feature Selection: SelectPercentile (chi-square, top 5%)
7. Random Forest Classification
8. Evaluation với classification report

## 📝 License

MIT License

## 👤 Author

lequangduyet03 - [GitHub](https://github.com/lequangduyet03)

## 🤝 Contributing

Contributions, issues và feature requests đều được chào đón!

## 🚧 Future Improvements

### Job Classifier
- Thu thập thêm data cho rare classes
- Thử nghiệm với XGBoost, Neural Networks
- Feature engineering nâng cao (years of experience, salary range)
- Hyperparameter tuning cho Random Forest
- Ensemble methods

---

⭐ Nếu project này hữu ích, hãy cho 1 star nhé!