import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

# 대용량 train.csv -> train.parquet 으로 변환환
# train = pd.read_csv('../data/raw/train.csv')
# train.to_parquet("../data/raw/train.parquet", compression="snappy")
# train.sample(10, random_state=42).to_csv("../data/sample_train.csv", index=False)

# 데이터 불러오기기
train = pd.read_parquet('../data/raw/train.parquet')
#train.head(5)

test = pd.read_csv('../data/raw/test.csv')
#test.head(5)

submission = pd.read_csv('../data/raw/sample_submission.csv')
#submission.head(5)


# 데이터 정보
train.columns
train.shape   # (6995056, 3)
test.shape   # (1747689, 2)

# 데이터 확인 - http로 시작하는 게 있는지 확인인
train[train['URL'].str[:4] == 'http'] # 1564개 있음음


# 악성 URL 확인 : 특징 파악
train[train['label']==1].head(50)
train['label'].value_counts(normalize=True)
train[train['URL'].str.contains(r'[A-Z]', regex=True)]["label"].value_counts(normalize=True)
.groupby['label'].agg(count_label=('label','count')) #.value_counts()
train[train['label']==0].head(50)

# -------------------------------------
# 대문자 여부 파생 변수 근거

# 대문자 데이터 확인 
train['has_capital'] = train['URL'].str.contains(r'[A-Z]', regex=True)
train[train['URL'].str.contains(r"http(?!s)", case=False, regex=True)]
pd.crosstab(train['has_http'], train['label'], margins=True)


# P(대문자|악성) = P(악성&대문자) / P(악성)
MC_p = np.sum((train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==1))/train.shape[0]
M_p = np.sum(train['label']==1)/train.shape[0]
CbarM = MC_p / M_p  # 0.15548115946289115

# P(대문자|정상) = P(정상&대문자) / P(정상)
NC_p = np.sum((train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==0))/train.shape[0]
N_p = np.sum(train['label']==0)/train.shape[0]
CbarN = NC_p / N_p   # 0.0017728762638442078

# P(악성|대문자) = P(악성&대문자) / P(대문자)
C_p = np.sum(train['URL'].str.contains(r'[A-Z]', regex=True))/train.shape[0]
MC_p = np.sum((train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==1))/train.shape[0]
MbarC = MC_p / C_p   # 0.9619394399440181   => 이걸 보니 대문자를 근거로 악성이라고 분류해도 되지 않을까? 그래서 대문자 여부 파생변수를 추가하기.

# P(정상|대문자) = P(정상&대문자) / P(대문자)
C_p = np.sum(train['URL'].str.contains(r'[A-Z]', regex=True))/train.shape[0]
NC_p = np.sum((train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==0))/train.shape[0]
NbarC = NC_p / C_p   # 0.03806056005598188


대문자 하나만 있으면 정상인가?
train[(train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==0)]


# ---------------------------------------
# http 여부 파생 변수 근거 : 의미 있음음

# http만 데이터 확인 (https 말고) 
train['has_http'] = train['URL'].str.contains(r"http(?!s)", case=False, regex=True)
train[train['URL'].str.contains(r"http(?!s)", case=False, regex=True)]
pd.crosstab(train['has_http'], train['label'], margins=True)

# P(http|악성) = P(악성&http) / P(악성)
Mhttp_p = np.sum(train['URL'].str.contains(r"http(?!s)", case=False, regex=True)&(train['label']==1))/train.shape[0]
M_p = np.sum(train['label']==1)/train.shape[0]
httpbarM = Mhttp_p / M_p  # 0.0020512532134702795

# P(http|정상) = P(정상&http) / P(정상)
Nhttp_p = np.sum(train['URL'].str.contains(r"http(?!s)", case=False, regex=True)&(train['label']==0))/train.shape[0]
N_p = np.sum(train['label']==0)/train.shape[0]
httpbarN = Nhttp_p / N_p   # 0.00008084477820999349

# P(악성|http) = P(악성&http) / P(http)
http_p = np.sum(train['URL'].str.contains(r"http(?!s)", case=False, regex=True))/train.shape[0]
Mhttp_p = np.sum(train['URL'].str.contains(r"http(?!s)", case=False, regex=True)&(train['label']==1))/train.shape[0]
Mbarhttp = Mhttp_p / http_p   # 0.8796930665935873   => 이걸 보니 http를 근거로 악성이라고 분류해도 되지 않을까? 그래서 http 여부 파생변수를 추가하기.

# P(악성|nothttp) = P(악성&nothttp) / P(nothttp)
nothttp_p = np.sum(~train['URL'].str.contains(r"http(?!s)", case=False, regex=True))/train.shape[0]
Mnothttp_p = np.sum(~train['URL'].str.contains(r"http(?!s)", case=False, regex=True)&(train['label']==1))/train.shape[0]
Mbarnothttp = Mnothttp_p / nothttp_p   # 0.22337234836993466


# P(정상|http) = P(정상&http) / P(http)
http_p = np.sum(train['URL'].str.contains(r"http(?!s)", case=False, regex=True))/train.shape[0]
Nhttp_p = np.sum(train['URL'].str.contains(r"http(?!s)", case=False, regex=True)&(train['label']==0))/train.shape[0]
Nbarhttp = Nhttp_p / http_p   # 0.12030693340641271

# P(정상|nothttp) = P(정상&nothttp) / P(nothttp)
nothttp_p = np.sum(~train['URL'].str.contains(r"http(?!s)", case=False, regex=True))/train.shape[0]
Nnothttp_p = np.sum(~train['URL'].str.contains(r"http(?!s)", case=False, regex=True)&(train['label']==0))/train.shape[0]
Nbarnothttp = Nnothttp_p / nothttp_p  # 0.7766276516300653


# P(악성|http)  0.879  >  P(악성|nothttp) 0.223  
# P(정상|http) 0.120  <  P(정상|nothttp) 0.776 
# => http가 있으면 악성일 가능성이 높고, http가 없으면 정상일 가능성이 높음.

