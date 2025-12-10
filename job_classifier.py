import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from imblearn.over_sampling import RandomOverSampler

# Tiền xử lý Location
def adjust_location(location):
    result = re.findall("\,\s[A-Z]{2}", location)
    if len(result) > 0:
        return result[0][2:]
    else:
        return location

# Load và Clean dữ liệu
data = pd.read_excel("final_project.ods", dtype=str)
data = data.dropna(axis=0)

# Tách Features và Target
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
x["location"] = x["location"].apply(adjust_location)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)
print(y_train.value_counts())
print("-----")

# Over-sampling (xử lý imbalanced data)
over_sampler = RandomOverSampler(random_state=42)
x_train, y_train = over_sampler.fit_resample(x_train, y_train)
print(y_train.value_counts())


# Feature Engineering Pipeline
preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(), "title"),
    ("desc", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=0.01, max_df=0.99), "description"),
    ("others", OneHotEncoder(handle_unknown="ignore"), ["location", "function", "industry"]),
])


# Model Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    # ("feature_selector", SelectKBest(chi2, k=500)),
    ("feature_selector", SelectPercentile(chi2, percentile=5)),
    ("classifier", RandomForestClassifier(random_state=100))
])

#Training và Evaluation
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))

