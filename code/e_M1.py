import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from lightgbm import LGBMClassifier





# 데이터 불러오기
# train = pd.read_parquet('../data/raw/train.parquet')
#train.head(5)


train = pd.read_csv('../data/sample_train.csv')
#train.head(5)

test = pd.read_csv('../data/sample_test.csv')
#test.head(5)




train2 = train.copy()
train2['URL'] = train2['URL'].str.replace('[.]','.')
train2['URL'] = train2['URL'].str.strip("'")

X = train2['URL']
y = train2['label']


# TF-IDF
vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(3, 5), max_features=50000)  
X_tfidf = vectorizer.fit_transform(X) 

features = vectorizer.get_feature_names_out()
print(features)	

X_tfidf.shape


# LGBM 학습
lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42,
    n_jobs=-1
)


results = cross_validate(
    lgbm,
    X_tfidf,
    y,
    cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True
)

results.keys()

# 평균 성능 출력
print("fit_time: ", results['fit_time'])
results['test_accuracy']
results['test_precision']

print("평균 accuracy:", results['test_accuracy'].mean())
print("평균 precision:", results['test_precision'].mean())
print("평균 recall:", results['test_recall'].mean())
print("평균 F1-score:", results['test_f1'].mean())

# 평균 accuracy: 0.925202
# 평균 precision: 0.9466019676899802
# 평균 recall: 0.7038793103448275
# 평균 F1-score: 0.8073841267336566




# 4. 예측 및 평가
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]




model.fit(X_train, y_train)
	