# 1번 : [sample] TF-IDF + LGBM_cv5 + class_weight='balanced'
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer 
from imblearn.metrics import geometric_mean_score
from lightgbm import LGBMClassifier


# 데이터 불러오기
train_raw = pd.read_csv('../data/sample_train.csv')
#train_raw.head(5)

test_raw = pd.read_csv('../data/sample_test.csv')
#test_raw.head(5)




train = train_raw.copy()
train['URL'] = train['URL'].str.replace('[.]','.')
train['URL'] = train['URL'].str.strip("'")
train['URL'] = train['URL'].str.lower()

test = test_raw.copy()
test['URL'] = test['URL'].str.replace('[.]','.')
test['URL'] = test['URL'].str.strip("'")
test['URL'] = test['URL'].str.lower()



train_X = train['URL']
train_y = train['label']
test_X = test['URL']


# TF-IDF
vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(3, 5), max_features=50000)  
X_train_tfidf = vectorizer.fit_transform(train_X) 
X_test_tfidf = vectorizer.transform(test_X)   

features = vectorizer.get_feature_names_out()
print(features)	

X_train_tfidf.shape
X_test_tfidf.shape



# LGBM 학습
lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)


results = cross_validate(
    lgbm,
    X_train_tfidf,
    train_y,
    cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring={
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'gmean': make_scorer(geometric_mean_score)
    },
    return_train_score= False
)


# 평균 성능 출력
print("평균 fit_time(초):", results['fit_time'].mean())
print("평균 score_time(초):", results['score_time'].mean())
print("평균 accuracy:", results['test_accuracy'].mean())
print("평균 precision:", results['test_precision'].mean())
print("평균 recall:", results['test_recall'].mean())
print("평균 F1-score:", results['test_f1'].mean())
print("평균 roc_auc:", results['test_roc_auc'].mean())
print("평균 gmean:", results['test_gmean'].mean())



# 기본 + class_weight='balanced' 설정 안 했을 때
# 평균 fit_time(초): 145.95274481773376
# 평균 score_time(초): 1.1471296310424806
# 평균 accuracy: 0.925408
# 평균 precision: 0.9469224244379711
# 평균 recall: 0.7045887212643678
# 평균 F1-score: 0.807967006165139
# 평균 roc_auc: 0.9408496850931023
# 평균 gmean: 0.8346301818624966


# 기본 + class_weight='balanced' 설정
# 평균 fit_time(초): 148.8505084514618
# 평균 score_time(초): 1.372413730621338
# 평균 accuracy: 0.9087419999999999
# 평균 precision: 0.7766959010795815
# 평균 recall: 0.8285290948275861
# 평균 F1-score: 0.8017601942910128
# 평균 roc_auc: 0.9407077365836487
# 평균 gmean: 0.8786113096460724




# ----------------  성능을 확인한 모델 그대로 다시 전체 훈련 데이터 학습
lgbm.fit(X_train_tfidf, train_y)  # class_weight='balanced' 설정



# ----------------  예측
y_prob = lgbm.predict_proba(X_test_tfidf)[:, 1]
y_pred = lgbm.predict(X_test_tfidf)

test_prob = test_raw.copy()
test_prob['label'] = y_prob

test_pred = test_raw.copy()
test_pred['label'] = y_pred

# test_prob.to_csv('../data/test/e_M1_prob.csv', index=False)
# test_pred.to_csv('../data/test/e_M1_pred.csv', index=False)
