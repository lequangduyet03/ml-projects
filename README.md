# Machine Learning Projects - Classification, Regression & Time Series

Dự án Machine Learning với 5 bài toán: Phân loại bệnh tiểu đường, Dự đoán điểm số học sinh, Phân loại cấp độ nghề nghiệp, và 2 bài toán Time Series dự đoán nồng độ CO2.

## 📋 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Kết quả](#kết-quả)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)

## 🎯 Giới thiệu

Repository này chứa 5 dự án Machine Learning:

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

### 4. Time Series - Dự đoán CO2 (Direct Multi-step)
- **Thuật toán**: Linear Regression (Multi-output)
- **Mục tiêu**: Dự đoán nồng độ CO2 cho 3 tuần tiếp theo cùng lúc
- **Dataset**: co2.csv (Time series data)
- **Phương pháp**: Direct Multi-step Forecasting
- **Window size**: 5 tuần
- **Target size**: 3 tuần
- **Đặc điểm**: Train 3 models riêng biệt cho mỗi bước dự đoán

### 5. Time Series - Dự đoán CO2 (Recursive)
- **Thuật toán**: Linear Regression
- **Mục tiêu**: Dự đoán nồng độ CO2 cho nhiều tuần tiếp theo
- **Dataset**: co2.csv (Time series data)
- **Phương pháp**: Recursive Forecasting
- **Window size**: 5 tuần
- **Đặc điểm**: 
  - Sử dụng 1 model duy nhất
  - Dự đoán từng bước, mỗi dự đoán được dùng làm input cho bước tiếp theo
  - Có visualization so sánh train/test/prediction

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

### CO2 Time Series Dataset
- **time**: Timestamp (datetime)
- **co2**: Nồng độ CO2 trong khí quyển
- **Đặc điểm**: 
  - Time series data với missing values (đã xử lý bằng interpolation)
  - Train/test split theo thời gian (80/20)
  - Window-based features (sliding window)

## 🛠️ Công nghệ sử dụng

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib (visualization)
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

### Chạy Time Series - Direct Multi-step
```bash
python direct_ts.py
```

Output:
- Metrics (MAE, MSE, R²) cho 3 models (dự đoán tuần 1, 2, 3)
- So sánh hiệu suất giữa các bước dự đoán

### Chạy Time Series - Recursive Forecasting
```bash
python recursive_ts.py
```

Output:
- Dự đoán 10 tuần tiếp theo từ dữ liệu ban đầu
- Hiển thị MAE, MSE, R² score
- Visualization: đồ thị so sánh train/test/prediction
- Minh họa quá trình recursive prediction

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

### Time Series - Direct Multi-step Model
Dự đoán 3 tuần cùng lúc với 3 models riêng biệt:
- **Model 1** (tuần +1): MAE, MSE, R² varies
- **Model 2** (tuần +2): MAE, MSE, R² varies
- **Model 3** (tuần +3): MAE, MSE, R² varies
- **Ưu điểm**: Mỗi model được tối ưu cho bước dự đoán cụ thể
- **Nhược điểm**: Cần train nhiều models, không có dependency giữa các predictions

### Time Series - Recursive Model
- **MAE**: 0.36
- **MSE**: 0.22
- **R² Score**: 0.99 ✨
- **Ưu điểm**: 
  - Chỉ cần 1 model
  - Có thể dự đoán vô hạn bước về tương lai
  - R² score rất cao (~99%)
- **Nhược điểm**: 
  - Lỗi tích lũy qua các bước
  - Uncertainty tăng theo thời gian dự đoán

## 📁 Cấu trúc thư mục

```
ml-projects/
│
├── classification.py           # Code phân loại bệnh tiểu đường
├── regression.py              # Code dự đoán điểm số
├── job_classifier.py          # Code phân loại cấp độ nghề nghiệp
├── direct_ts.py               # Code time series - direct multi-step
├── recursive_ts.py            # Code time series - recursive forecasting
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

### Time Series Pipeline

#### Direct Multi-step Approach:
1. Load và preprocess data (interpolate missing values)
2. Create sliding windows:
   - Input: 5 tuần liên tiếp (window_size=5)
   - Output: 3 tuần tiếp theo (target_size=3)
3. Train/Test split theo thời gian (80/20)
4. Train 3 models riêng biệt:
   - Model 1: predict t+1
   - Model 2: predict t+2
   - Model 3: predict t+3
5. Evaluate từng model độc lập

#### Recursive Approach:
1. Load và preprocess data (interpolate missing values)
2. Create sliding windows:
   - Input: 5 tuần liên tiếp (window_size=5)
   - Output: 1 tuần tiếp theo
3. Train/Test split theo thời gian (80/20)
4. Train 1 model duy nhất
5. Recursive prediction:
   - Dự đoán tuần tiếp theo
   - Thêm prediction vào input window
   - Loại bỏ giá trị cũ nhất
   - Lặp lại cho các tuần sau
6. Visualization kết quả

## 📊 So sánh các phương pháp Time Series

| Tiêu chí | Direct Multi-step | Recursive |
|----------|------------------|-----------|
| **Số models** | 3 models | 1 model |
| **Complexity** | Cao hơn | Đơn giản hơn |
| **Training time** | Lâu hơn | Nhanh hơn |
| **Error accumulation** | Không có | Có (tăng theo thời gian) |
| **Flexibility** | Giới hạn (3 bước) | Vô hạn bước |
| **Best for** | Short-term forecast | Long-term forecast |
| **R² Score** | Varies | 0.99 |

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

### Time Series Models
- Thử nghiệm với ARIMA, SARIMA
- Implement LSTM/GRU cho sequential data
- Thêm seasonal decomposition
- Hybrid models (ARIMA + ML)
- Add confidence intervals cho predictions
- Feature engineering: lag features, rolling statistics, trend components
- Thử nghiệm với Random Forest Regressor thay vì Linear Regression

---

⭐ Nếu project này hữu ích, hãy cho 1 star nhé!