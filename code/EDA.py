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

#http?
#  