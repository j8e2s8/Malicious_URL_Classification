# 📦 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from collections import Counter
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import precision_recall_curve
from scipy.sparse import hstack
from xgboost import XGBClassifier
import optuna

# 📁 2. 데이터 불러오기 및 소문자 변환
df_train = pd.read_csv("../data/raw/train.csv")
df_test = pd.read_csv("../data/raw/test.csv")
df_train['URL'] = df_train['URL'].str.lower()
df_test['URL'] = df_test['URL'].str.lower()

# ❌ 3. 결측치 및 이상치 제거
df_train = df_train.dropna(subset=['URL'])
df_train = df_train[df_train['URL'].str.len() > 1].reset_index(drop=True)


# 🎯 4. URL 특징 추출 함수
def extract_url_features(url):
    try:
        parsed = urlparse(url)
        features = {}

        features['url_len'] = len(url)
        features['domain_len'] = len(parsed.netloc)
        features['path_len'] = len(parsed.path)
        features['query_len'] = len(parsed.query)

        special_chars = ['@', '?', '-', '_', '=', '&', '%', '.']
        for char in special_chars:
            features[f'char_{char}'] = url.count(char)

        num_digits = sum(c.isdigit() for c in url)
        features['digit_ratio'] = num_digits / len(url) if len(url) > 0 else 0

        suspicious_keywords = ['secure', 'update', 'login', 'verify', 'bank', 'online']
        features['has_suspicious_keyword'] = int(any(k in url for k in suspicious_keywords))

        counter = Counter(url)
        probabilities = [count / len(url) for count in counter.values()]
        features['entropy'] = entropy(probabilities, base=2)

        features['num_subdomains'] = parsed.netloc.count('.')

    except Exception:
        features = {
            'url_len': 0, 'domain_len': 0, 'path_len': 0, 'query_len': 0,
            'digit_ratio': 0, 'has_suspicious_keyword': 0, 'entropy': 0, 'num_subdomains': 0,
            'char_@': 0, 'char_?': 0, 'char_-': 0, 'char__': 0,
            'char_=': 0, 'char_&': 0, 'char_%': 0, 'char_.': 0
        }

    return pd.Series(features)

# 🧪 5. 샘플링 (빠른 학습용)
df_train_sample = df_train.sample(n=100000, random_state=42).reset_index(drop=True)

# 🔧 6. 특징 추출 및 벡터화
url_features_train = df_train_sample['URL'].apply(extract_url_features)

vectorizer_ngram = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=1000)
X_train_ngram = vectorizer_ngram.fit_transform(df_train_sample['URL'])

vectorizer_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=1000)
X_train_tfidf = vectorizer_tfidf.fit_transform(df_train_sample['URL'])

X_train_final = hstack([X_train_ngram, X_train_tfidf, url_features_train.values])
y_train = df_train_sample['label']

# ✂️ 7. 데이터 분할
X_tr, X_val, y_tr, y_val = train_test_split(X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42)

# 🚀 8. Optuna 튜닝
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'tree_method': 'hist',
        'eval_metric': 'logloss',
        'random_state': 42,
        'use_label_encoder': False,
        'n_jobs': -1
    }
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("📌 Best Parameters:", study.best_params)

# 📊 9. 최적 모델로 재학습 및 평가
xgb_best = XGBClassifier(**study.best_params, tree_method='hist', eval_metric='logloss', random_state=42, use_label_encoder=False, n_jobs=-1)
xgb_best.fit(X_tr, y_tr)

# 예측 및 threshold 조정
y_proba_best = xgb_best.predict_proba(X_val)[:, 1]
threshold = 0.4
y_pred_best = (y_proba_best >= threshold).astype(int)

# 결과 출력
print(f"📌 Classification Report (threshold = {threshold}):")
print(classification_report(y_val, y_pred_best))
print("📌 Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_best))
print("📌 ROC AUC:", roc_auc_score(y_val, y_proba_best))

# 🔁 10. 다양한 threshold 비교
for t in [0.5, 0.4, 0.3, 0.25]:
    y_pred = (y_proba_best >= t).astype(int)
    print(f"\nThreshold = {t}")
    print(classification_report(y_val, y_pred, digits=4))
