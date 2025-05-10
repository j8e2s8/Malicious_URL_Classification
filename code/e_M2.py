# 2번 : [sample] 전체 url TF-IDF + LGBM_cv5 (임계값 결정) + class_weight='balanced'
# auc-roc = 0.940708  (임계값 0.6)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from lightgbm import LGBMClassifier



# 데이터 불러오기
df_raw = pd.read_csv('../data/sample_train.csv')
#train_raw.head(5)

test_raw = pd.read_csv('../data/sample_test.csv')
#test_raw.head(5)



# 데이터 전처리
df = df_raw.copy()
df['URL'] = df['URL'].str.replace('[.]','.')
df['URL'] = df['URL'].str.strip("'")
df['URL'] = df['URL'].str.lower()

test = test_raw.copy()
test['URL'] = test['URL'].str.replace('[.]','.')
test['URL'] = test['URL'].str.strip("'")
test['URL'] = test['URL'].str.lower()



df_X = df['URL']
df_y = df['label']

test_X = test['URL']


# TF-IDF
vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(3, 5), max_features=50000)  
df_X_tfidf = vectorizer.fit_transform(df_X) 
test_X_tfidf = vectorizer.transform(test_X)

features = vectorizer.get_feature_names_out()
print(features)	

df_X_tfidf.shape


# LGBM 학습
lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)


# threshold_df 만드는 함수
def evaluate_thresholds(y_true, y_prob):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from imblearn.metrics import geometric_mean_score

    thresholds=np.arange(0.0, 1.01, 0.1)
    results = []

    for t in thresholds:
        y_pred = (y_prob > t).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0) # zero_division=0 : 분모가 0이면 0으로 반환해라
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        g_mean = geometric_mean_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # .ravel() : 행렬 형태를 한 줄로 만들어줌.

        predicted_pos = (y_pred == 1).sum()
        actual_pos = (y_true == 1).sum()
        predicted_neg = (y_pred == 0).sum()
        actual_neg = (y_true == 0).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results.append({
            'Threshold': round(t, 2),
            'Predicted class 1 N': predicted_pos,
            'Actual class 1 N': actual_pos,
            'Predicted class 0 N': predicted_neg,
            'Actual class 0 N': actual_neg,
            'Accuracy' : round(accuracy, 6),
            'Precision': round(precision, 6),
            'Recall': round(recall, 6),
            'F1 Score': round(f1, 6),
            'G-mean' : round(g_mean, 6),
            'FPR': round(fpr, 6),
            'ROC AUC': round(auc, 6)
        })
    df_result = pd.DataFrame(results).sort_values('Threshold', ascending=False)
    return df_result


# fold별 임계값 평가지표 구해서 임계값 결정하기
threshold_df_list = []
skf = StratifiedKFold(n_splits=5 , shuffle=True, random_state=42)
							# n_split : fold 개수  , shuffle=True : 데이터를 분할 할 때 타겟 클래스 비율을 각 fold마다 동일하게 유지


for train_index, valid_index in skf.split(df_X_tfidf, df_y):    # df를 n_splits 개수 fold로 train과 valid 인덱스 나눠줌.
    train_X, valid_X = df_X_tfidf[train_index] , df_X_tfidf[valid_index]
    train_y, valid_y = df_y[train_index] , df_y[valid_index]
	
	# 직접 나눈 train set으로 모델 학습시키고 valid set으로 평가지표 구해서 검증하기
    lgbm.fit(train_X, train_y)
    y_prob = lgbm.predict_proba(valid_X)[:, 1] 
	
    threshold_df_list.append(evaluate_thresholds(valid_y, y_prob))


threshold_df_all = pd.concat(threshold_df_list, axis=0)
mean_threshold = threshold_df_all.groupby('Threshold').mean().reset_index().sort_values('Threshold', ascending=False)


threshold_optimal = float(mean_threshold[mean_threshold['F1 Score'] == mean_threshold['F1 Score'].max()]['Threshold'])
valid_auc_mean = mean_threshold['ROC AUC'].unique()
print(f"valid_auc_mean : {valid_auc_mean}") # [0.940708]



# ----------------  성능을 확인한 모델 그대로 다시 전체 훈련 데이터 학습
lgbm.fit(df_X_tfidf, df_y)  # class_weight='balanced' 설정



# ----------------  예측
y_prob = lgbm.predict_proba(test_X_tfidf)[:, 1]
y_pred = (y_prob > threshold_optimal).astype(int)



test_pred = test_raw.copy()
test_pred['label'] = y_pred


# test_pred.to_csv('../data/test/e_M2_pred.csv', index=False)


