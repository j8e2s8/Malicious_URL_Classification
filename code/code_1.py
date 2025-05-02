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
train_raw = pd.read_csv("../data/raw/train.csv")
test_raw = pd.read_csv("../data/raw/test.csv")

train = train_raw.copy()
test =  test_raw.copy()

train["iscapital"] = train['URL'].str.contains(r'[A-Z]', regex=True)
test["iscapital"] = test['URL'].str.contains(r'[A-Z]', regex=True).astype(int)

train['URL'] = train['URL'].str.lower()
test['URL'] = test['URL'].str.lower()
train['URL'] = train['URL'].str.replace("[.]",".")
test['URL'] = train['URL'].str.replace("[.]",".")

train['URL'] = train['URL'].str.strip("'")
test['URL'] = test['URL'].str.strip("'")


train[train['URL'].str.contains(r'http(?!s)', case=False)] 
test[test['URL'].str.contains(r'http(?!s)', case=False)]

# ❌ 3. 결측치 및 이상치 제거
train = train.dropna(subset=['URL'])
test = test.dropna(subset=['URL'])  
train = train[train['URL'].str.len() > 1].reset_index(drop=True)
test = test[test['URL'].str.len() > 1].reset_index(drop=True)   

# 🎯 4. URL 특징 추출 함수
def extract_url_features(url):
    try:
        parsed = urlparse(url)

        features["path_num"] = parsed.path.str.count(r'[/]')
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
            "dash_num" : 0,'url_len': 0, 'domain_len': 0, 'path_len': 0, 'query_len': 0,
            'digit_ratio': 0, 'has_suspicious_keyword': 0, 'entropy': 0, 'num_subdomains': 0,
            'char_@': 0, 'char_?': 0, 'char_-': 0, 'char__': 0,
            'char_=': 0, 'char_&': 0, 'char_%': 0, 'char_.': 0
        }

    return pd.Series(features)

# 🧪 5. 샘플링 (빠른 학습용)
train_sample = train.sample(n=100000, random_state=42).reset_index(drop=True)

# 🔧 6. 특징 추출 및 벡터화
url_features_train = train_sample['URL'].apply(extract_url_features)

vectorizer_ngram = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=1000)
X_train_ngram = vectorizer_ngram.fit_transform(train_sample['URL'])

vectorizer_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=1000)
X_train_tfidf = vectorizer_tfidf.fit_transform(train_sample['URL'])

X_train_final = hstack([X_train_ngram, X_train_tfidf, url_features_train.values])
y_train = train_sample['label']

pd.DataFrame(X_train_final.toarray())

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

"""
import re
from urllib.parse import urlparse
from collections import Counter
from scipy.stats import entropy
import ipaddress

# 확장된 URL 특징 추출 함수
def extract_url_features_enhanced(url):
    try:
        parsed = urlparse(url)
        features = {}

        # 기본 구조적 특징
        features['url_len'] = len(url)
        features['domain_len'] = len(parsed.netloc)
        features['path_len'] = len(parsed.path)
        features['query_len'] = len(parsed.query)
        features['subdomain_count'] = parsed.netloc.count('.') - 1  # 보통 도메인+TLD는 제외

        # 문자 기반 특징
        special_chars = ['@', '?', '-', '_', '=', '&', '%', '.', '/', ':']
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_special_chars'] = sum(url.count(c) for c in special_chars)
        features['has_uppercase'] = int(any(c.isupper() for c in url))
        features['digit_ratio'] = features['num_digits'] / len(url) if len(url) > 0 else 0

        # 보안 여부
        features['is_https'] = int(parsed.scheme == 'https')

        # 키워드 포함 여부
        phishing_keywords = ['login', 'signin', 'secure', 'account', 'verify', 'update', 'confirm', 'password', 'reset', 'validate', 'auth', 'webscr']
        features['has_phishing_keyword'] = int(any(k in url.lower() for k in phishing_keywords))

        # IP 기반 여부
        try:
            host = parsed.netloc.split(':')[0]
            ipaddress.ip_address(host)
            features['is_ip_url'] = 1
        except:
            features['is_ip_url'] = 0

        # Shannon Entropy
        counter = Counter(url)
        probabilities = [count / len(url) for count in counter.values()] if len(url) > 0 else [0]
        features['entropy'] = entropy(probabilities, base=2) if len(url) > 0 else 0

        # 최상위 도메인 (TLD)
        tld_match = re.search(r"\.([a-z]{2,10})(\/|$)", url)
        features['tld'] = tld_match.group(1) if tld_match else 'unknown'

    except Exception as e:
        features = {
            'url_len': 0, 'domain_len': 0, 'path_len': 0, 'query_len': 0,
            'subdomain_count': 0, 'num_digits': 0, 'num_special_chars': 0,
            'has_uppercase': 0, 'digit_ratio': 0, 'is_https': 0,
            'has_phishing_keyword': 0, 'is_ip_url': 0, 'entropy': 0, 'tld': 'unknown'
        }

    return features

url_features_train = train_sample['URL'].apply(lambda x: pd.Series(extract_url_features_enhanced(x)))
url_features_train = pd.get_dummies(url_features_train, columns=['tld'], drop_first=True)
X_train_final = hstack([X_train_ngram, X_train_tfidf, url_features_train.values])
"""