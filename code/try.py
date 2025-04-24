import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
train[train['URL'].str[:4] == 'http'] # 1564개 있음
train['label'].value_counts(normalize=True)  # 정상 URL : 0.776285 , 악성 URL : 0.223715 => 불균형 데이터터


# 악성 URL 확인 : 특징 파악
train[train['label']==1].head(50)
train[train['label']==0].head(50)


# -------------------------------------
# 대문자 여부 파생 변수 근거 : 의미 있음

# 대문자 데이터 확인 
train['has_capital'] = train['URL'].str.contains(r'[A-Z]', regex=True)
train[train['URL'].str.contains(r'[A-Z]', regex=True)]
pd.crosstab(train['has_capital'], train['label'], margins=True)


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
MbarC = MC_p / C_p   # 0.9619394399440181  

# P(악성|not대문자) = P(악성&not대문자) / P(not대문자)
notC_p = np.sum(~train['URL'].str.contains(r'[A-Z]', regex=True))/train.shape[0]
MnotC_p = np.sum((~train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==1))/train.shape[0]
MbarnotC = MnotC_p / notC_p   # 0.1960192918633717


# P(정상|대문자) = P(정상&대문자) / P(대문자)
C_p = np.sum(train['URL'].str.contains(r'[A-Z]', regex=True))/train.shape[0]
NC_p = np.sum((train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==0))/train.shape[0]
NbarC = NC_p / C_p   # 0.03806056005598188

# P(정상|not대문자) = P(정상&not대문자) / P(not대문자)
notC_p = np.sum(~train['URL'].str.contains(r'[A-Z]', regex=True))/train.shape[0]
NnotC_p = np.sum((~train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==0))/train.shape[0]
NbarnotC = NnotC_p / notC_p   # 0.8039807081366283

# P(악성|대문자)  0.961  >  P(악성|not대문자) 0.196  
# P(정상|대문자) 0.038  <  P(정상|not대문자) 0.803 
# => 대문자가 있으면 악성일 가능성이 높고, 대문자가 없으면 정상일 가능성이 높음.



# ----------------------------
# 대문자 개수 파생 변수 근거 : 의미 있음 (하지만 빼자)
# 대문자 개수가 적으면 정상 URL일까? 기준 : 대문자 10개
train[(train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==0)]
C_df = train.copy()
C_df['count_C'] = train['URL'].str.count(r'[A-Z]')
C_df['count_C'].unique()

len(C_df['count_C'].unique())

C_df['over_10'] = train['URL'].str.count(r'[A-Z]')>=2
pd.crosstab(C_df['over_10'], C_df['label'], margins=True)



# P(C10|악성) = P(악성&C10) / P(악성)
MC10_p = np.sum((C_df['over_10']==1)&(train['label']==1))/train.shape[0]
M_p = np.sum(train['label']==1)/train.shape[0]
C10barM = MC10_p / M_p  # 0.03191200443224059

# P(C10|정상) = P(정상&C10) / P(정상)
NC10_p = np.sum((C_df['over_10']==1)&(train['label']==0))/train.shape[0]
N_p = np.sum(train['label']==0)/train.shape[0]
C10barN = NC10_p / N_p   # 0.0002756825352627796

# P(악성|C10) = P(악성&C10) / P(C10)
C10_p = np.sum((C_df['over_10']==1))/train.shape[0]
MC10_p = np.sum((C_df['over_10']==1)&(train['label']==1))/train.shape[0]
MbarC10 = MC10_p / C10_p   # 0.9708958705964694  

# P(악성|notC10) = P(악성&notC10) / P(notC10)
notC10_p = np.sum((~C_df['over_10']==1))/train.shape[0]
MnotC10_p = np.sum((~C_df['over_10']==1)&(train['label']==1))/train.shape[0]
MbarnotC10 = MnotC10_p / notC10_p   # 0.21817985431230397


# P(정상|C10) = P(정상&C10) / P(C10)
C10_p = np.sum((C_df['over_10']==1))/train.shape[0]
NC10_p = np.sum((C_df['over_10']==1)&(train['label']==0))/train.shape[0]
NbarC10 = NC10_p / C10_p   # 0.0291041294035306

# P(정상|notC10) = P(정상&notC10) / P(notC10)
notC10_p = np.sum((~C_df['over_10']==1))/train.shape[0]
NnotC10_p = np.sum((~C_df['over_10']==1)&(train['label']==0))/train.shape[0]
NbarnotC10 = NnotC10_p / notC10_p  # 0.7818201456876961


# P(악성|C10)  0.970  >  P(악성|notC10) 0.218  
# P(정상|C10) 0.029  <  P(정상|notC10) 0.781 
# => C10가 있으면 악성일 가능성이 높고, C10가 없으면 정상일 가능성이 높음.

# P(악성|대문자)  0.961  >  P(악성|not대문자) 0.196  
# P(정상|대문자) 0.038  <  P(정상|not대문자) 0.803 
# => 대문자가 있으면 악성일 가능성이 높고, 대문자가 없으면 정상일 가능성이 높음.



# 고민 : 대문자 여부 파생 변수 vs 대문자 10개 이상 여부 파생 변수 둘 중 하나만 남겨놔야할까?
# 판단 : 대문자 여부 파생 변수만 남겨놓자.
# 이유1 : 대문자를 가지고 있는 데이터의 수는 252939개인데, 대문자 10개 이상인 데이터의 수는 51436밖에 안됨.
#       51436개에서 악성 URL 분류하는 것보다는 더 많은 데이터인 252939개에서 분류하는게 성능 신뢰도가 있을 것 같음.
# 이유2 : 밑에 groupby와 시각화 자료를 봤을 때, 대문자 0개일 때의 악성 URL 비율은 0.2 이고 그 외 대문자 개수마다 악성 URL 비율은 1에 가까움. 
#        대문자 개수 파생변수보다는 대문자 여부 이진 파생변수가 더 의미 있을 것 같음.

# groupby와 시각화로 확인해보기 : (대문자 0개제외) 대문자가 몇 개든간에 악성 비율은 거의 1에 가까움
# 대문자가 0개일 때는 악성 비율이 0.2임.
group_M = C_df.groupby('count_C', as_index=False).agg(악성_수=('label','sum'))
group_M1 = C_df.groupby('count_C', as_index=False).agg(전체=('label','count'))
group_M = pd.merge(group_M, group_M1, how='left', on='count_C')
group_M['악성_비율'] = group_M['악성_수']/group_M['전체']

len(C_df[C_df['count_C']==1])  # 대문자가 1개인 데이터가 74503개인데 그 중 악성이 72416개임.

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('대문자 개수별 악성URL 비율', fontsize=12, pad=20)
plt.xlabel('대문자 개수', fontsize=12)  # x축 제목과 글씨 크기 지정
plt.ylabel('악성URL 비율', fontsize=12)  # y축 제목과 글씨 크기 지정
sns.barplot(data=group_M, x='count_C', y='악성_비율')




# ---------------------------------------
# http 여부 파생 변수 근거 : 의미 있음

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
Mbarhttp = Mhttp_p / http_p   # 0.8796930665935873  

# P(악성|nothttp) = P(악성&nothttp) / P(nothttp)
nothttp_p = np.sum(~train['URL'].str.contains(r"http(?!s)", case=False, regex=True))/train.shape[0]
Mnothttp_p = np.sum((~train['URL'].str.contains(r"http(?!s)", case=False, regex=True))&(train['label']==1))/train.shape[0]
Mbarnothttp = Mnothttp_p / nothttp_p   # 0.22337234836993466


# P(정상|http) = P(정상&http) / P(http)
http_p = np.sum(train['URL'].str.contains(r"http(?!s)", case=False, regex=True))/train.shape[0]
Nhttp_p = np.sum(train['URL'].str.contains(r"http(?!s)", case=False, regex=True)&(train['label']==0))/train.shape[0]
Nbarhttp = Nhttp_p / http_p   # 0.12030693340641271

# P(정상|nothttp) = P(정상&nothttp) / P(nothttp)
nothttp_p = np.sum(~train['URL'].str.contains(r"http(?!s)", case=False, regex=True))/train.shape[0]
Nnothttp_p = np.sum((~train['URL'].str.contains(r"http(?!s)", case=False, regex=True))&(train['label']==0))/train.shape[0]
Nbarnothttp = Nnothttp_p / nothttp_p  # 0.7766276516300653


# P(악성|http)  0.879  >  P(악성|nothttp) 0.223  
# P(정상|http) 0.120  <  P(정상|nothttp) 0.776 
# => http가 있으면 악성일 가능성이 높고, http가 없으면 정상일 가능성이 높음.

