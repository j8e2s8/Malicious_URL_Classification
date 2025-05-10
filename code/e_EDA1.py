import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from urllib.parse import urlparse
pd.set_option('display.max_colwidth', None)

# 대용량 train.csv -> train.parquet 으로 변환환
# train = pd.read_csv('../data/raw/train.csv')
# train.to_parquet("../data/raw/train.parquet", compression="snappy")
# train.sample(500000, random_state=42).to_csv("../data/sample_train.csv", index=False)
# test = pd.read_csv('../data/raw/test.csv')
# test.sample(125000, random_state=42).to_csv("../data/sample_test.csv", index=False)

# train_parsed.sample(500000, random_state=42).to_csv('../data/sample_eda_train.csv', index=False)
# test_parsed.sample(125000, random_state=42).to_csv('../data/sample_eda_test.csv', index=False)



# ------------------------------- 데이터 불러오기
# train 원본
train_raw = pd.read_parquet('../data/raw/train.parquet')
#train.head(5)

# eda_train
results_train=[]
for i in np.arange(14):
	chunk = pd.read_csv(f"../data/eda_train{i}.csv", index_col=0, keep_default_na=False)
	results_train.append(chunk)
	print(i)
train_parsed = pd.concat(results_train, ignore_index=True)

# test 원본
test_raw = pd.read_csv('../data/raw/test.csv')
#test.head(5)

# eda_test
results_test=[]
for i in np.arange(4):
	chunk = pd.read_csv(f"../data/eda_test{i}.csv", index_col=0, keep_default_na=False)
	results_test.append(chunk)
	print(i)
test_parsed = pd.concat(results_test, ignore_index=True)



# ---------------------------- train set 결정된 전처리
# 데이터 전처리
train = train_raw.copy()
train['URL'] = train['URL'].str.replace('[.]','.')
train['URL'] = train['URL'].str.strip("'")


# 대문자 여부 파생변수
train['is_capital'] = train['URL'].str.contains(r'[A-Z]', regex=True)

# 악성일 가능성이 높은 요소들의 True False 파생변수
# . or - or _ 로 IP 스타일, 인코딩, 포트, 공백, 영어 외 언어, 유효한 특수문자 (! , @ , # , % , $ , & , * , ( , ) , _ , + , - , = , [ , ] , ' , \ , : , " , , , / , ? , `)
train['is_factor'] = train['URL'].str.contains(r'\b\d{1,3}(\.\d{1,3}){3}\b|\b\d{1,3}(\-\d{1,3}){3}\b|\b\d{1,3}(\_\d{1,3}){3}\b', regex=True) \
						| train['URL'].str.contains(r'[^a-zA-Z0-9\n\t\^{};|<>~\.]', regex=True )


# 특수문자 수 파생변수
train['count_sc'] = train['URL'].str.count(r'[!@#$%&*()_+\-=\[\]\'\\:",?`]')


# http 여부 파생변수
train['is_http'] = train['URL'].str.contains(r"http(?!s)", case=False, regex=True)

# 데이터 전처리2
train['URL'] = train['URL'].str.lower()


# https:// 채워주기
def complete_url(url):
	from urllib.parse import urlparse
	parsed = urlparse(url)
	if 'protected' in url:
		if '/' not in url.split('protected')[0]:
			pass
		elif '/' in url.split('protected')[0]:
			url = 'https://' + url
	elif '://' in url:
		if '/' in url.split('://')[0] or '.' in url.split('://')[0] or '?' in url.split('://')[0]:
			url = 'https://' + url
	elif ':' in url:
		if '/' in url.split(':')[0] or '.' in url.split(':')[0] or '?' in url.split(':')[0]:
			url = 'https://' + url
		elif parsed.scheme =='':
			url = 'https://' + url
	elif '://' not in url:
		url = 'https://' + url
	return url


train['complete_URL'] = train['URL'].apply(complete_url)



# parse 하기
def safe_urlparse(url):
	from urllib.parse import urlparse
	try:
		parsed = urlparse(url)
		if (parsed.scheme=='') & ('://' in url):
			final_scheme = url.split('://')[0]
			url = url.replace(final_scheme, 'https')
			parsed = urlparse(url)
		else:
			final_scheme = parsed.scheme
		return pd.Series({
			'scheme' : final_scheme,
			'netloc' : parsed.netloc,
			'username' : parsed.username,
			'password' : (
						str(int(parsed.password)) if parsed.password is not None and isinstance(parsed.password, (int, float)) 
						else '' if parsed.password is None 
						else str(parsed.password)
					),
			'hostname' : parsed.hostname,
			'port' : (
						str(int(parsed.port)) if parsed.port is not None and isinstance(parsed.port, (int, float)) 
						else '' if parsed.port is None 
						else str(parsed.port)
					),
			'path' : parsed.path,
			'query' : parsed.query,
			'fragment' : parsed.fragment })
	except Exception:
		return pd.Series({
			'scheme' : 'exception',
			'netloc' : 'exception',
			'username' : 'exception',
			'password' : 'exception',
			'hostname' : 'exception',
			'port' : 'exception',
			'path' : 'exception',
			'query' : 'exception',
			'fragment' : 'exception' })




# 데이터 셋을 조금씩 분할해서 urlparse하고 다시 합치기
chunks = np.array_split(train, 14)  # 50만 개씩 나누기 (700만 / 14 ≈ 50만)
results = []
for i, chunk in enumerate(chunks):
	parsed = chunk['complete_URL'].apply(safe_urlparse)
	chunk = pd.concat([chunk.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
	print(i)
	results.append(chunk)

train_parsed = pd.concat(results, ignore_index=True)



# 경로 수 파생변수 
def count_path(path):
	if (path == '') or (path == 'exception'):
		return 0
	return len([i for i in path.split('/') if i])

train_parsed['count_path'] = train_parsed['path'].apply(count_path)



# 쿼리 길이 파생변수
train_parsed['len_query'] = train_parsed['query'].str.len()


# eda_train 데이터 저장하기
# chunks = np.array_split(train_parsed, 14)
# for i, chunk in enumerate(chunks):
# 	chunk.to_csv(f"../data/eda_train{i}.csv")
# 	print(i)






# ---------------------------- test set 결정된 전처리
# 데이터 전처리
test = test_raw.copy()
test['URL'] = test['URL'].str.replace('[.]','.')
test['URL'] = test['URL'].str.strip("'")


# 대문자 여부 파생변수
test['is_capital'] = test['URL'].str.contains(r'[A-Z]', regex=True)

# 악성일 가능성이 높은 요소들의 True False 파생변수
# . or - or _ 로 IP 스타일, 인코딩, 포트, 공백, 영어 외 언어, 유효한 특수문자 (! , @ , # , % , $ , & , * , ( , ) , _ , + , - , = , [ , ] , ' , \ , : , " , , , / , ? , `)
test['is_factor'] = test['URL'].str.contains(r'\b\d{1,3}(\.\d{1,3}){3}\b|\b\d{1,3}(\-\d{1,3}){3}\b|\b\d{1,3}(\_\d{1,3}){3}\b', regex=True) \
						| test['URL'].str.contains(r'[^a-zA-Z0-9\n\t\^{};|<>~\.]', regex=True )


# 특수문자 수 파생변수
test['count_sc'] = test['URL'].str.count(r'[!@#$%&*()_+\-=\[\]\'\\:",?`]')


# http 여부 파생변수
test['is_http'] = test['URL'].str.contains(r"http(?!s)", case=False, regex=True)

# 데이터 전처리2
test['URL'] = test['URL'].str.lower()


# https:// 채워주기
def complete_url(url):
	from urllib.parse import urlparse
	parsed = urlparse(url)
	if 'protected' in url:
		if '/' not in url.split('protected')[0]:
			pass
		elif '/' in url.split('protected')[0]:
			url = 'https://' + url
	elif '://' in url:
		if '/' in url.split('://')[0] or '.' in url.split('://')[0] or '?' in url.split('://')[0]:
			url = 'https://' + url
	elif ':' in url:
		if '/' in url.split(':')[0] or '.' in url.split(':')[0] or '?' in url.split(':')[0]:
			url = 'https://' + url
		elif parsed.scheme =='':
			url = 'https://' + url
	elif '://' not in url:
		url = 'https://' + url
	return url


test['complete_URL'] = test['URL'].apply(complete_url)



# parse 하기
def safe_urlparse(url):
	from urllib.parse import urlparse
	try:
		parsed = urlparse(url)
		if (parsed.scheme=='') & ('://' in url):
			final_scheme = url.split('://')[0]
			url = url.replace(final_scheme, 'https')
			parsed = urlparse(url)
		else:
			final_scheme = parsed.scheme
		return pd.Series({
			'scheme' : final_scheme,
			'netloc' : parsed.netloc,
			'username' : parsed.username,
			'password' : (
						str(int(parsed.password)) if parsed.password is not None and isinstance(parsed.password, (int, float)) 
						else '' if parsed.password is None 
						else str(parsed.password)
					),
			'hostname' : parsed.hostname,
			'port' : (
						str(int(parsed.port)) if parsed.port is not None and isinstance(parsed.port, (int, float)) 
						else '' if parsed.port is None 
						else str(parsed.port)
					),
			'path' : parsed.path,
			'query' : parsed.query,
			'fragment' : parsed.fragment })
	except Exception:
		return pd.Series({
			'scheme' : 'exception',
			'netloc' : 'exception',
			'username' : 'exception',
			'password' : 'exception',
			'hostname' : 'exception',
			'port' : 'exception',
			'path' : 'exception',
			'query' : 'exception',
			'fragment' : 'exception' })


# 데이터 셋을 조금씩 분할해서 urlparse하고 다시 합치기
chunks = np.array_split(test, 14)  # 50만 개씩 나누기 (700만 / 14 ≈ 50만)
results = []
for i, chunk in enumerate(chunks):
	parsed = chunk['complete_URL'].apply(safe_urlparse)
	chunk = pd.concat([chunk.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1) 
	print(i)
	results.append(chunk)

test_parsed = pd.concat(results, ignore_index=True)




# 경로 수 파생변수 
def count_path(path):
	if (path == '') or (path == 'exception'):
		return 0
	return len([i for i in path.split('/') if i])

test_parsed['count_path'] = test_parsed['path'].apply(count_path)



# 쿼리 길이 파생변수
test_parsed['len_query'] = test_parsed['query'].str.len()


# eda_test 데이터 저장하기
# chunks = np.array_split(test_parsed, 4)
# for i, chunk in enumerate(chunks):
# 	chunk.to_csv(f"../data/eda_test{i}.csv")
# 	print(i)




# -------------------------   데이터 확인하기 + 가설검정
def Derive(df, A, text):
	print("해당 URL 확인 :\n",df[A][['URL', 'label']].head(10),"\n", "-"*40) 
	c= pd.crosstab(A, df['label'], margins=True)
	print("교차표 확인 :\n", c,"\n")
	print(f"{text}에 해당하는 데이터 {c.iloc[1,2]}개 -> 정상 {c.iloc[1,0]}개, 악성 {c.iloc[1,1]}개\n", "-"*40)

	# P(악성|text) = P(악성&text) / P(text)
	text_p = np.sum(A)/df.shape[0]
	Mtext_p = np.sum(A &(df['label']==1))/df.shape[0]
	Mbartext = Mtext_p / text_p    

	# P(악성|nottext) = P(악성&nottext) / P(nottext)
	nottext_p = np.sum(~A)/df.shape[0]
	Mnottext_p = np.sum((~A)&(df['label']==1))/df.shape[0]
	Mbarnottext = Mnottext_p / nottext_p   

	# P(정상|text) = P(정상&text) / P(text)
	text_p = np.sum(A)/df.shape[0]
	Ntext_p = np.sum(A&(df['label']==0))/df.shape[0]
	Nbartext = Ntext_p / text_p   

	# P(정상|nottext) = P(정상&nottext) / P(nottext)
	nottext_p = np.sum(~A)/df.shape[0]
	Nnottext_p = np.sum((~A)&(df['label']==0))/df.shape[0]
	Nbarnottext = Nnottext_p / nottext_p  

	if Mbartext > Mbarnottext:
		print(f"P(악성|{text})  {Mbartext: .3f}  >  P(악성|not{text}) {Mbarnottext: .3f}")
	elif Mbartext < Mbarnottext:
		print(f"P(악성|{text})  {Mbartext: .3f}  <  P(악성|not{text}) {Mbarnottext: .3f}")
	
	if Nbartext < Nbarnottext:
		print(f"P(정상|{text}) {Nbartext: .3f}  <  P(정상|not{text}) {Nbarnottext: .3f}\n","-"*40)
	elif Nbartext > Nbarnottext:
		print(f"P(정상|{text}) {Nbartext: .3f}  >  P(정상|not{text}) {Nbarnottext: .3f}\n","-"*40)
	# P(악성|text) > P(악성|nottext) 이면서 # P(정상|text) < P(정상|nottext) 이면,
	# text이면 악성일 가능성이 높고, text가 아니면 정상일 가능성이 높아서 text True False 파생변수 만드는게 의미 있을 거임.

	from scipy.stats import chi2_contingency
	from scipy.stats import fisher_exact

	c = pd.crosstab(A, df['label'])
	chi2, p, dof, expected = chi2_contingency(c)
	
	if (expected < 5).any():
		print("예상 빈도 중 5 미만인 값이 있습니다.")
		oddsratio, p = fisher_exact(c)
		if p < 0.05:
			print("귀무가설 기각: 독립변수와 타겟 변수는 독립적이지 않다. 관련이 있다.")
			print(f"Odds Ratio: {oddsratio:.4f}")
		else:
			print("귀무가설 채택: 독립변수와 타겟 변수는 독립적이다. 관련이 없다")

	else:
		print("모든 예상 빈도가 5 이상입니다.")
		if p < 0.05:
			print("귀무가설 기각: 독립변수와 타겟 변수는 독립적이지 않다. 관련이 있다.")
			n = c.values.sum() # 분석에 사용할 데이터 수가 아니라 교차표에 사용한 데이터 수임 (필터링된 데이터 수)
			min_dim = min(c.shape) - 1  
			cramers_v = np.sqrt(chi2 / (n * min_dim))
			print(f"Cramer's V: {cramers_v:.4f} 정도의 연관이 있다")
			vc = df[A]['label'].value_counts(normalize=True)
			print(f"전체 데이터 {df.shape[0]}개에 비해, False는 {vc[0]:.3f} 비율 , True는 {vc[1]:.3f} 비율")
		else:
			print("귀무가설 채택: 독립변수와 타겟 변수는 독립적이다. 관련이 없다")
			

	

# 데이터 정보
train.columns
train.shape   # (6995056, 3)
test.shape   # (1747689, 2)

# 정상 URL : 0.776285 , 악성 URL : 0.223715 => 불균형 데이터
train['label'].value_counts(normalize=True)  

# 결측치는 없음.
pd.isna(train).sum()


# 악성 URL 확인 : 특징 파악
train[train['label']==1].head(50)
train[train['label']==0].head(50)




# -------------------------------------
# 1. 대문자 여부 파생 변수 근거 : 의미 있음

# 대문자 데이터 확인 : 252939개 -> 정상 9627개, 악성 243312개
# 카이제곱 검정 : 관련 있다 -> 0.3431 정도의 연관이 있다

# P(악성|C)   0.962  >  P(악성|notC)  0.196
# P(정상|C)  0.038  <  P(정상|notC)  0.804
# => 대문자가 있으면 악성일 가능성이 높고, 대문자가 없으면 정상일 가능성이 높음.

A = train['URL'].str.contains(r'[A-Z]', regex=True)
Derive(train,A, 'C')

# False : 0.964 비율 , True : 0.036 비율   (이 정도 비율은 Cramér's V 값이 괜찮게 나오나봄)
a = train['URL'].str.contains(r'[A-Z]', regex=True).value_counts(normalize=True)
print(f"False : {a[0]:.3f} 비율 , True : {a[1]:.3f} 비율")




# ----------------------------
# 2. 대문자 개수 파생 변수 c근거 : 의미 있음 (하지만 빼자)
# 대문자 개수가 적으면 정상 URL일까? 기준 : 대문자 10개
train[(train['URL'].str.contains(r'[A-Z]', regex=True))&(train['label']==0)]
C_df = train.copy()
C_df['count_C'] = train['URL'].str.count(r'[A-Z]')
C_df['count_C'].unique()

len(C_df['count_C'].unique())

C_df['over_10'] = train['URL'].str.count(r'[A-Z]')>=2
pd.crosstab(C_df['over_10'], C_df['label'], margins=True)


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




# --------------------------------
# 3. (연관성이 미미해서 합칠) 악성일 가능성이 높은 요소들의 True False 파생변수


# 공백, 영어 외 언어 : 11794개 -> 정상 140개, 악성 11654개
# 카이 제곱 검정 : 관련이 있다 -> 0.0754 정도의 연관이 있다
# P(악성|factor)   0.989  >  P(악성|notfactor)  0.222
# P(정상|factor)  0.011  <  P(정상|notfactor)  0.778
A= train['URL'].str.contains(r'[^a-zA-Z0-9\n\t!@#$%^&*()_+\-=\[\]{};\'\\:"|,<>?`~\./]', regex=True )
Derive(train, A, 'factor')
# 전체 데이터 6995056개에 비해, False는 0.012 비율 , True는 0.988 비율


# 공백, 영어 외 언어, 유효한 특수문자 (! , @ , # , % , $ , & , * , ( , ) , _ , + , - , = , [ , ] , ' , \ , : , " , , , / , ? , `)
# factor에 해당하는 데이터 2932170개 -> 정상 1661583개, 악성 1270587개
# 카이제곱 검정 : 관련이 있다. -> 0.4273 정도의 연관이 있다
# P(악성|factor)   0.433  >  P(악성|notfactor)  0.072
# P(정상|factor)  0.567  <  P(정상|notfactor)  0.928
A= train['URL'].str.contains(r'[^a-zA-Z0-9\n\t^{};|<>~\.]', regex=True )
Derive(train, A, 'factor')


# . or - or _ 로 IP 스타일, 공백, 영어 외 언어, 유효한 특수문자 (! , @ , # , % , $ , & , * , ( , ) , _ , + , - , = , [ , ] , ' , \ , : , " , , , / , ? , `)
# factor에 해당하는 데이터 2938450개 -> 정상 1661779개, 악성 1276671개
# 카이제곱 검정 : 관련이 있다. -> 0.4304 정도의 연관이 있다
# P(악성|factor)   0.434  >  P(악성|notfactor)  0.071
# P(정상|factor)  0.566  <  P(정상|notfactor)  0.929
A= train['URL'].str.contains(r'\b\d{1,3}(\.\d{1,3}){3}\b|\b\d{1,3}(\-\d{1,3}){3}\b|\b\d{1,3}(\_\d{1,3}){3}\b', regex=True) \
	| train['URL'].str.contains(r'[^a-zA-Z0-9\n\t\^{};|<>~\.]', regex=True )
Derive(train, A, 'factor')
# 전체 데이터 6995056개에 비해, False는 0.566 비율 , True는 0.434 비율



# http 파생변수 만든다고 크게 달라지는 건 없는 듯 <- 빼야 하나 (파생변수 의미만 희석되는 듯?)
# http , . or - or _ 로 IP 스타일, 공백, 영어 외 언어, 유효한 특수문자 (! , @ , # , % , $ , & , * , ( , ) , _ , + , - , = , [ , ] , ' , \ , : , " , , , / , ? , `)
# factor에 해당하는 데이터 2938765개 -> 정상 1661881개, 악성 1276884개
# 카이제곱 검정 : 관련이 있다. -> 0.4305 정도의 연관이 있다
# P(악성|factor)   0.434  >  P(악성|notfactor)  0.071
# P(정상|factor)  0.566  <  P(정상|notfactor)  0.929
A= train['URL'].str.contains(r'\b\d{1,3}(\.\d{1,3}){3}\b|\b\d{1,3}(\-\d{1,3}){3}\b|\b\d{1,3}(\_\d{1,3}){3}\b', regex=True) \
	| train['URL'].str.contains(r'[^a-zA-Z0-9\n\t^{};|<>~\.]', regex=True ) \
	| train['URL'].str.contains(r"http(?!s)", case=False, regex=True)
Derive(train, A, 'factor')
# 전체 데이터 6995056개에 비해, False는 0.566 비율 , True는 0.434 비율




# 참고 ***********
# IP에 해당하는 데이터 33185개 -> 정상 305개, 악성 32880개
# 카이제곱 검정 : 관련이 있다 -> 0.1271 정도의 연관이 있다
# P(악성|IP)   0.991  >  P(악성|notIP)  0.220
# P(정상|IP)  0.009  <  P(정상|notIP)  0.780
A = train['URL'].str.contains(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', regex=True)
Derive(train, A, 'IP')

# 포트가 명시된 url은 악성 가능이 높음 (이건 TF-IDF가 해결해주지 않을까 싶음)
train[train['URL'].str.contains(r'\b\:\d{1,5}\b', regex=True)& (train['label']==0)]
train[train['URL'].str.contains(r'\b\:\d{1,5}\b', regex=True)]['label'].value_counts()



# . or - or _ 로 IP 스타일에 해당하는 데이터 41542개 -> 정상 409개, 악성 41133개
# _ 형식은 실제 IP 스타일을 따라한 건 아닌것같지만, 악성이 많음
# 카이제곱 검정 : 관련이 있다 -> 0.1422 정도의 연관이 있다
#P(악성|IPstyle)   0.990  >  P(악성|notIPstyle)  0.219
#P(정상|IPstyle)  0.010  <  P(정상|notIPstyle)  0.781
train[train['URL'].str.contains(r'\b\d{1,3}(\.\d{1,3}){3}\b|\b\d{1,3}(\-\d{1,3}){3}\b|\b\d{1,3}(\_\d{1,3}){3}\b', regex=True)]
A= train['URL'].str.contains(r'\b\d{1,3}(\.\d{1,3}){3}\b|\b\d{1,3}(\-\d{1,3}){3}\b|\b\d{1,3}(\_\d{1,3}){3}\b', regex=True)
Derive(train, A, 'IPstyle')




# 공백 : 공백이 포함된 URL 11692개 -> 정상 133개, 악성 11559개
# 카이제곱 검정 : 관련있다 -> 0.0751 정도의 연관이 있다
# P(악성|s)  0.989  >  P(악성|nots) 0.222 
# P(정상|s) 0.011  <  P(정상|nots) 0.778 
# => s가 있으면 악성일 가능성이 높고, s가 없으면 정상일 가능성이 높음.
train[train['URL'].str.contains(r'[\s]', regex=True )]
A = train['URL'].str.contains(r'[\s]', regex=True )
Derive(train, A, 's')


# 줄바꿈 : 줄바꿈이 포함된 URL 1개 -> 정상 0개, 악성 1개 (의미 없음)
# <br>로 되어 있는 것도 있지만, 2개 (정상) 밖에 없음. 
# 피셔 정확 검정 : 관련 없음
# P(악성|enter)   1.000  >  P(악성|notenter)  0.224
# P(정상|enter)  0.000  <  P(정상|notenter)  0.776
# => 줄바꿈가 있으면 악성일 가능성이 높고, 줄바꿈가 없으면 정상일 가능성이 높음.
A = train['URL'].str.contains(r'[\n]', regex=True )
Derive(train, A, 'enter')


# 탭 : 탭이 포함된 URL 2개 -> 정상 0개, 악성 2개 (의미 없음)
# 피셔 정확 검정 : 관련 없음
# P(악성|tab)   1.000  >  P(악성|nottab)  0.224
# P(정상|tab)  0.000  <  P(정상|nottab)  0.776
# => 탭가 있으면 악성일 가능성이 높고, 탭가 없으면 정상일 가능성이 높음.
A = train['URL'].str.contains(r'[\t]', regex=True )
Derive(train, A, 'tab')


# 영어 외 언어.일반적이지 않은 특수 문자 : 영어가 아닌 다른 언어가 포함된 URL 111개 -> 정상 9개, 악성 102개
# 영어 외 언어.일반적이지 않은 특수 문자 = 영어, 공백, 줄바꿈 ,탭, 모든 일반 특수 문자가 아닌 문자임.
# 카이제곱 검정 : 관련 있다 -> 0.0066 정도의 연관이 있다
# P(악성|ol)   0.919  >  P(악성|notol)  0.224
# P(정상|ol)  0.081  <  P(정상|notol)  0.776
# => ol가 있으면 악성일 가능성이 높고, ol가 없으면 정상일 가능성이 높다.
A = train['URL'].str.contains(r'[^a-zA-Z0-9\s\n\t!@#$%^&*()_+\-=\[\]{};\'\\:"|,<>?`~\./]+', regex=True )
train[A]
Derive(train, A, 'ol')


# 인코딩 데이터 : 21968개 -> 정상 4286개, 악성 17682개
# 카이제곱 검정 : 관련이 있다. -> 0.0783 정도의 연관이 있다
# P(악성|encode)   0.805  >  P(악성|notencode)  0.222
# P(정상|encode)  0.195  <  P(정상|notencode)  0.778
A = train['URL'].str.contains(r'%[a-fA-F0-9]{2}', regex=True)
train[A]
Derive(train, A, 'encode')

A = train['URL'].str.contains(r'%20', regex=True)
train[A]
Derive(train, A, '%20')

A = train['URL'].str.contains(r'%[a-fA-F0-9]{2}', regex=True) & ~train['URL'].str.contains(r'%20', regex=True)
train[A]
Derive(train, A, 'e')


# 모든 특수 문자를 포함 여부 True False는 의미가 없는 파생변수임.
# ./ 포함 모든 특수문자 : 6994954개 -> 정상 5430078개, 악성 1564876개
# 카이제곱 검정 : 관련 없다
# P(악성|allsc)   0.224  >  P(악성|notallsc)  0.206
# P(정상|allsc)  0.776  <  P(정상|notallsc)  0.794
A = train['URL'].str.contains(r'[!@#$%^&*()_+\-=\[\]{};\'\\:"|,<>?`~\./]+', regex=True )
Derive(train, A, 'allsc')
# 모든 특수문자에 대해서 하나라도 없는 데이터 (정상 많음음)
train[~train['URL'].str.contains(r'[!@#$%^&*()_+\-=\[\]{};\'\\:"|,<>?`~\./]+', regex=True )]


# 특수문자 (.  / 제외) : 1601694개 -> 정상 898408개, 악성 703286개
# 카이 제곱 검정 : 관련 있다 -> 0.2816 정도의 연관이 있다 (근데 조건부 확률이 별로로)
# P(악성|sc)  0.439  >  P(악성|notsc) 0.159 
# P(정상|sc) 0.560  <  P(정상|notsc) 0.840
# => sc가 있으면 악성일 가능성이 높고, sc가 없으면 정상일 가능성이 높긴 하지만, 크게 차이나는 건 아님.
A = train['URL'].str.contains(r'[!@#$%^&*()_+\-=\[\]{};\'\\:"|,<>?`~]+', regex=True )
Derive(train, A, 'sc')
A = train['URL'].str.contains(r'[^\w./]', regex=True )
Derive(train, A, 'sc')





# 전체 특수 문자 : !@#$%^&*()_+\-=\[\]{};\'\\:"|,<>?`~\./
import re
sc_list = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_' , '+', '-', '=', '[', ']', '{', '}', ';', "'", '\\', ':', '"', '|', ',', '<', '.', '>', '/', '?', '`', '~']

for i in sc_list:
	print(f"{i}에 대한 :")
	if i == '\\':
		A= train['URL'].str.contains(re.escape(i), regex=True )
		Derive(train, A, 'c')
		print("-"*50,"\n")
	else:
		A= train['URL'].str.contains(rf'\{i}', regex=True )
		Derive(train, A, 'c')
		print("-"*50,"\n")


# => 유효한 특수 문자 결론 : ! , @ , # , % , $ , & , * , ( , ) , _ , + , - , = , [ , ] , ' , \ , : , " , , , / , ? , `
# 비유효한 특수 문자 : ^, { , } , ; , | , < , > , . , ~
# 유효한 특수문자 데이터 : 2932150개 -> 정상 1661577개, 악성 1270573개
# P(악성|factor)   0.433  >  P(악성|notfactor)  0.072
# P(정상|factor)  0.567  <  P(정상|notfactor)  0.928
# 카이제곱 검정 : 관련이 있다. -> 0.4273 정도의 연관이 있다
A= train['URL'].str.contains(r'[\!\@\#\%\$\&\*\(\)\_\+\-\=\[\]\'\:\"\,\/\?\`]', regex=True) \
	| train['URL'].str.contains(re.escape('\\'), regex=True)
Derive(train, A, 'factor')


# cramer's V = 0.0029 이정도면 아무런 관계가 없는거임. (모든 데이터에 있어도 이 정도 값은 나옴옴)


# ! : 1232개 -> 정상 236개, 악성 996개
# 0.0186 정도의 연관이 있다

# @ : 15377개 -> 정상 416개, 악성 14961개
# 0.0844 정도의 연관이 있다

# # : 7774개 -> 정상 1742개, 악성 6032개
# 0.0442 정도의 연관이 있다

# $ : 6539개 -> 정상 118개, 악성 6421개
# 0.0557 정도의 연관이 있다

# % : 22084개 -> 정상 4298개, 악성 17786개 (인코딩 %이든 특수문자 %이든 악성과 연관 있음)
# 0.0785 정도의 연관이 있다

# URL 인코딩(악성 가능성 높음)하면 %가 많이 생김 
# 인코딩 제외 특수문자 % 포함만 : 116개 -> 정상 12개, 악성 104개
# 0.0065 정도의 연관이 있다 (데이터가 적어서 그렇지 %도 연관이 있긴 한 듯)
A = train['URL'].str.contains(r'\%', regex=True ) & ~train['URL'].str.contains(r'%[0-9A-Fa-f]{2}', regex=True )
Derive(train, A, '%')
# 인코딩만 : 21968개 -> 정상 4286개, 악성 17682개
# 0.0783 정도의 연관이 있다 (인코딩 연관이 있긴 한듯)
train[train['URL'].str.contains(r'%[0-9A-Fa-f]{2}', regex=True )]
A = train['URL'].str.contains(r'%[0-9A-Fa-f]{2}', regex=True )
Derive(train, A, 'encode')

# ^ : 48개 -> 정상 26개, 악성 22개
# 0.0014 정도의 연관이 있다
train[train['URL'].str.contains('\^')] # ^ 는 무슨 의미가 있는거지? : '초기값이다', '모든 값이다', '특별히 escape 처리를 해야한다' 등 특별한 약속을 해버린 경우가 있을 수도 있음


# & : 86064개 -> 정상 14949개, 악성 71115개
# 0.1614 정도의 연관이 있다

# * : 2203개 -> 정상 310개, 악성 1893개
# 0.0271 정도의 연관이 있다
train[train['URL'].str.contains('\*')]
train[train['URL'].str.contains('\*')]['label'].value_counts()
train[train['URL'].str.contains('\*') & (~train['URL'].str.endswith('*'))]['label'].value_counts()


# ( : 2612개 -> 정상 459개, 악성 2153개
# 0.0278 정도의 연관이 있다
train[train['URL'].str.contains('\(')]
train[train['URL'].str.contains('\(')& ~train['URL'].str.contains('\)')]

# ) : 2565개 -> 정상 477개, 악성 2088개
# 0.0271 정도의 연관이 있다
train[train['URL'].str.contains('\)')]
train[train['URL'].str.contains('\)')& ~train['URL'].str.contains('\(')]

train[train['URL'].str.endswith(')')]

# _ : 175810개 -> 정상 81226개, 악성 94584개 (_ IP 이든 특수문자 _이든 악성과 연관 있음)
# 0.1211 정도의 연관이 있다
# _ IP 스타일로도 쓰임.
train[train['URL'].str.contains('\_')]

# _ IP 스타일 제외 특수문자 _ : 175746개 -> 정상 81223개, 악성 94523개
# 카이 제곱 검정 : 관련 있다 -> 0.1210 정도의 연관이 있다
# P(악성|_)   0.538  >  P(악성|not_)  0.216
# P(정상|_)  0.462  <  P(정상|not_)  0.784
# => _가 있으면 악성일 가능성이 높고, _가 없으면 정상일 가능성이 높긴 하지만, 크게 차이나는 건 아님.
A = train['URL'].str.contains(r'[_]', regex=True ) & ~train['URL'].str.contains(r'\b\d{1,3}(\_\d{1,3}){3}\b', regex=True)
Derive(train, A, '_')
# _ IP 스타일 데이터 64개 -> 정상 3개, 악성 61개
# 카이제곱 검정 : 관련이 있다. -> 0.0052 정도의 연관이 있다
# P(악성|_IPstyle)   0.953  >  P(악성|not_IPstyle)  0.224
# P(정상|_IPstyle)  0.047  <  P(정상|not_IPstyle)  0.776
A = train['URL'].str.contains(r'\b\d{1,3}(\_\d{1,3}){3}\b', regex=True)
Derive(train, A, '_IPstyle')
train[train['URL'].str.contains(r'\b\d{1,3}(\_\d{1,3}){3}\b', regex=True)]

# + : 4515개 -> 정상 1922개, 악성 2593개
# 0.0214 정도의 연관이 있다
train[train['URL'].str.contains('\+')]

# - : 1355195개 -> 정상 764812개, 악성 590383개 (전체 데이터가 악성이 워낙 적은데 -는 정상>악성 이어도 이정도는 악성일 가능성이 큰 거임임)
# Cramer's V: 0.2493 정도의 연관이 있다
train[train['URL'].str.contains('\-')]
A = train['URL'].str.contains(r'[-]', regex=True )
Derive(train, A, '-')
# - IP 스타일 제외 특수문자 - : 1346845개 -> 정상 764711개, 악성 582134개
# 카이 제곱 검정 : 관련 있다 -> 0.2443 정도의 연관이 있다
# P(악성|-)   0.432  >  P(악성|not-)  0.174
# P(정상|-)  0.568  <  P(정상|not-)  0.826
# => -가 있으면 악성일 가능성이 높고, -가 없으면 정상일 가능성이 높긴 하지만, 크게 차이나는 건 아님.
A = train['URL'].str.contains(r'[-]', regex=True ) & ~train['URL'].str.contains(r'\b\d{1,3}(\-\d{1,3}){3}\b', regex=True)
Derive(train, A, '-')
# - IP 스타일 데이터 8350개 -> 정상 101개, 악성 8249개
# 카이제곱 검정 : 관련이 있다. -> 0.0634 정도의 연관이 있다
# P(악성|-IPstyle)   0.988  >  P(악성|not-IPstyle)  0.223
# P(정상|-IPstyle)  0.012  <  P(정상|not-IPstyle)  0.777
A = train['URL'].str.contains(r'\b\d{1,3}(\-\d{1,3}){3}\b', regex=True)
Derive(train, A, '-IPstyle')
train[train['URL'].str.contains(r'\b\d{1,3}(\-\d{1,3}){3}\b', regex=True)]


# = : 189631개 -> 정상 42335개, 악성 147296개
# 0.2215 정도의 연관이 있다
train[train['URL'].str.contains('\=')]
train[train['URL'].str.contains('\=') & (train['label']==1)]

# [ : 11822개 -> 정상 17개, 악성 11805개
# 0.0765 정도의 연관이 있다
train[train['URL'].str.contains('\[') & ~train['URL'].str.contains('\]')]
# [email protected] 제외한 [ 데이터 : 27개 -> 정상 4개, 악성 23개
# 0.0029 정도의 연관이 있다
train[train['URL'].str.contains('\[') & ~train['URL'].str.contains('\]') & ~train['URL'].str.contains('protected')]
A = train['URL'].str.contains('\[') & ~train['URL'].str.contains('\]') & ~train['URL'].str.contains('protected')
Derive(train, A, '[')

# ] : 11846개 -> 정상 15개, 악성 11831개
# 0.0766 정도의 연관이 있다
# [email protected] 제외한 ] 데이터 : 51개 -> 정상 2개, 악성 49개
# 0.0047 정도의 연관이 있다
train[~train['URL'].str.contains('\[') & train['URL'].str.contains('\]') & ~train['URL'].str.contains('protected')]
A = ~train['URL'].str.contains('\[') & train['URL'].str.contains('\]') & ~train['URL'].str.contains('protected')
Derive(train, A, ']')

# [email protected] 외의 []만 악성과 연관 있을 줄 알았는데 그건 아님.
# [email protected] 데이터 : 11871개 -> 정상 54개, 악성 11817개
# 0.0763 정도의 연관이 있다
A = train['URL'].str.contains('protected')
Derive(train, A, 'protected')



# { : 42개 -> 정상 22개, 악성 20개
# 0.0014 정도의 연관이 있다 (이 정도는 관련 없는 듯)
train[train['URL'].str.contains('\{')]
train[train['URL'].str.contains('\{') & ~train['URL'].str.contains('\}')]

# } : 38개 -> 정상 22개, 악성 16개
# 0.0010 정도의 연관이 있다 (이 정도는 관련 없는 듯)
train[(~train['URL'].str.contains('\{')) & train['URL'].str.contains('\}')]
train[train['URL'].str.contains('\{|\}')]  # 변수의 의미, 미완성 의미 주려고
# 악성 {} : ex) {getemail} , {email} 같이 email을 입럭하라는 듯한 느낌임
# 정상 {} : ex) {11e02fda-6837-46f3-ab7e-dbba387f5b29}


# ; : 21467개 -> 정상 14624개, 악성 6843개 (구분자로 쓰여서 그런듯)
# 0.0127 정도의 연관이 있다 (이 정도는 관련 없는 듯)
train[train['URL'].str.contains('\;')]

# ' : 3825개 -> 정상 520개, 악성 3305개 ('가 대부분 url 앞 뒤에 붙어 있는 것으로 크롤링 같은거 해오다가 잘못 따라 붙어온 듯)
# 0.0359 정도의 연관이 있다
train[train['URL'].str.contains("\'")] # 3825개
# 'url'인거 데이터 3345개 -> 정상 265개, 악성 3080개 (데이터 셋 만드는 과정에서 잘못 가져왔나?)
# 0.0366 정도의 연관이 있다
train[train['URL'].str.startswith("'")]
train[train['URL'].str.startswith("'") | train['URL'].str.endswith("'")]['label'].value_counts()
A =train['URL'].str.startswith("'") | train['URL'].str.endswith("'")
Derive(train, A, "'url'")
# 'url'이 아닌 특수문자 ' : 480개 -> 정상 255개, 악성 225개
# 0.0049 정도의 연관이 있다
A = train['URL'].str.contains("\'") & ~train['URL'].str.startswith("'") & ~train['URL'].str.endswith("'")
Derive(train, A, "'")


# \ : 2723개 -> 정상 153개, 악성 2570개
# 0.0341 정도의 연관이 있다
train[train['URL'].str.contains(re.escape('\\'), regex=True)]

# : : 14936개 -> 정상 1658개, 악성 13278개
# 0.0738 정도의 연관이 있다
train[train['URL'].str.contains(':')]

# " : 3개 -> 정상 0개, 악성 3개
# 피셔 정확 검정 : 관련 있다 -> odds ratio : 무한 (매우 크게 관련 있다)
train[train['URL'].str.contains('"')]

# | : 213개 -> 정상 185개, 악성 28개 (구분자로 쓰여서 그런듯)
# 0.0012 정도의 연관이 있다
train[train['URL'].str.contains(r'\|',regex=True)]

# , : 2446개 -> 정상 133개, 악성 2313개
# 0.0324 정도의 연관이 있다
train[train['URL'].str.contains(',')]

# < : 8개 -> 정상 2개, 악성 6개
# 피셔 정확 검정 : 관련 있다 -> odds ratio : 10.41 (>랑 세트로 관련이 없다고 해야 하나?)
train[train['URL'].str.contains(r'\<', regex=True)]

# > : 159개 -> 정상 151개, 악성 8개 (<url>이랬는지 url>인게 많음. >제거해야 하지 않나? )
# 0.0019 정도의 연관이 있다
train[train['URL'].str.contains(r'\>', regex=True) ]
train[train['URL'].str.contains(r'\>', regex=True) & ~train['URL'].str.endswith('>')]#['label'].value_counts()
train[train['URL'].str.endswith('>')]['label'].value_counts()
# >로 끝나는 url 150개 중 147개가 정상 , 3개가 악성
# 근데 >로 안 끝나도 악성 5개 정상 4개임. >는 크게 영향 없는 듯
train[(~train['URL'].str.endswith('>')) & train['URL'].str.contains('>')]


# . : 6994883개 -> 정상 5430067개, 악성 1564816개 (구분자)
# 0.0029 정도의 연관이 있다 (이 정도는 관련 없음)

# / : 2114983개 -> 정상 1012096개, 악성 1102887개
# 0.4704 정도의 연관이 있다
train[train['URL'].str.contains('\/')]  # 경로가 많아 질 수록 악성인가? -> / 개수로 알아보기

# ? : 194443개 -> 정상 43458개, 악성 150985개
# 0.2243 정도의 연관이 있다
train[train['URL'].str.contains('\?')]

# ` : 7개 -> 정상 0개, 악성 7개
# 피셔 정확 검정 : 관련 있다 -> odds ratio : 무한 (매우 크게 관련 있음)
train[train['URL'].str.contains('\`')]

# ~ : 45032개 -> 정상 39076개, 악성 5956개 (서버 내에서 해당 사용자의 개인 웹 페이지 디렉토리를 의미)
# 0.0177 정도의 연관이 있다 (이 정도는 관련이 없는 듯)
train[train['URL'].str.contains('~')]

# mrqe.com(영화리뷰 사이트트) & =^ or ?^ & (연도) 가 거의 같은 url 안에 있음 
# (연도) 쓴 경우 : 37개 -> 정상 35개, 악성 2개 (mrqe.com이 아닌 다른 사이트임) <- 애매
# (연도)는 보통 =^, ?^ 가 같이 쓰인 듯. 그래서 ^가 cramer's V가 작게 나온 듯.
# (연도) 카이제곱 검정 : 관련 있다 -> 0.0009 정도의 연관이 있다
# P(악성|year)   0.054  <  P(악성|notyear)  0.224
# P(정상|year)  0.946  >  P(정상|notyear)  0.776
A = train['URL'].str.contains(r'\(\d{4}\)', regex=True)
Derive(train, A, 'year')
train[train['URL'].str.contains(r'\(\d{4}\)', regex=True)]
train[train['URL'].str.contains('\=\^') | train['URL'].str.contains('\?\^')]

# ***************






# [시각화] 유효한 특수문자가 많을 수록 정상보다 악성이 많아지는 경향임 
train['count_sc'] = train['URL'].str.count(r'[!@#$%&*()_+\-=\[\]\'\\:",?`]')
group_M = train.groupby('count_sc', as_index=False).agg(악성_수=('label','sum'))
group_M1 = train.groupby('count_sc', as_index=False).agg(전체=('label','count'))
group_M = pd.merge(group_M, group_M1, how='left', on='count_sc')
group_M['악성_비율'] = group_M['악성_수']/group_M['전체']

len(train[train['count_sc']==1])  # 특수문자가 1개인 데이터가 1022904개인데 그 중 악성이 313901개임.

plt.figure(figsize=(15,5))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('특수문자 개수별 악성URL 비율', fontsize=12, pad=20)
plt.xlabel('특수문자 개수', fontsize=12)  # x축 제목과 글씨 크기 지정
plt.ylabel('악성URL 비율', fontsize=12)  # y축 제목과 글씨 크기 지정
sns.barplot(data=group_M, x='count_sc', y='악성_비율')

#
plt.figure(figsize=(10,6))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('특수문자 개수별 악성.정상 개수', fontsize=12, pad=20)
plt.xlabel('특수문자 개수', fontsize=12)  
plt.ylabel('악성.정상 개수', fontsize=12)  
sns.countplot(data=train[train['count_sc']<=10], x='count_sc', hue='label')

#
plt.figure(figsize=(10,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('특수문자 개수별 악성.정상 개수', fontsize=12, pad=20)
plt.xlabel('특수문자 개수', fontsize=12)  
plt.ylabel('악성.정상 개수', fontsize=12)  
sns.countplot(data=train[train['count_sc']>10], x='count_sc', hue='label')



# [시각화] 경로 수 (/ 수)가 많을 수록 정상보다 악성이 많아짐
train['count_/'] = train['URL'].str.count(r'[/]')
group_M = train.groupby('count_/', as_index=False).agg(악성_수=('label','sum'))
group_M1 = train.groupby('count_/', as_index=False).agg(전체=('label','count'))
group_M = pd.merge(group_M, group_M1, how='left', on='count_/')
group_M['악성_비율'] = group_M['악성_수']/group_M['전체']

len(train[train['count_/']==1])  # 특수문자가 1개인 데이터가 1022904개인데 그 중 악성이 313901개임.

plt.figure(figsize=(15,5))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('경로 개수별 악성URL 비율', fontsize=12, pad=20)
plt.xlabel('경로 개수', fontsize=12)  # x축 제목과 글씨 크기 지정
plt.ylabel('악성URL 비율', fontsize=12)  # y축 제목과 글씨 크기 지정
sns.barplot(data=group_M, x='count_/', y='악성_비율')


#
plt.figure(figsize=(10,8))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('경로 개수별 악성.정상 개수', fontsize=12, pad=20)
plt.xlabel('경로 개수', fontsize=12)  
plt.ylabel('악성.정상 개수', fontsize=12)  
sns.countplot(data=train[train['count_/']<=10], x='count_/', hue='label')

#
plt.figure(figsize=(10,8))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('경로 개수별 악성.정상 개수', fontsize=12, pad=20)
plt.xlabel('경로 개수', fontsize=12)  
plt.ylabel('악성.정상 개수', fontsize=12)  
sns.countplot(data=train[train['count_/']>10], x='count_/', hue='label')






# ---------------------------------------
# 3. http 여부 파생 변수 근거 : 의미 있음 but 데이터 수가 너무 적어서 이렇게만 하면 의미가 없을 것 같음.
# 특별히 https 보다 http라고 해서 악성인 건 아닌 것 같음. 

train[train['URL'].str.contains(r'[A-Z]', regex=True)]

# 1) r"http(?!s)" 방법으로 조회한 http 포함 데이터 : 3649개 
# https가 있든 없는 http만 있으면 뽑아줌
train[train['URL'].str.contains(r"http(?!s)", case=False, regex=True)]

# 2) r"http" 있고, r"https 없는 방법으로 조회한 데이터 : 3303개
# https가 없으면서 http만 있으면 뽑아줌
train[train['URL'].str.contains(r"http", case=False, regex=True) & ~train['URL'].str.contains(r"https", case=False, regex=True)]

# 1)이면서 2)가 아닌 데이터 확인
train[train['URL'].str.contains(r"http(?!s)", case=False, regex=True) \
 & ~(train['URL'].str.contains(r"http", case=False, regex=True) & ~train['URL'].str.contains(r"https", case=False, regex=True))]['label'].value_counts()





train[train['URL'].str.contains('^https')] # http 포함이 아니라 http로 시작하는 제대로 된 url도 악성일까?



train[train['URL'].str.contains('^ftp', case=False)]['label'].value_counts()
train[train['URL'].str.contains('^http', case=False)] # 
train[train['URL'].str.contains('^H', case=True)]
train[train['URL'].str.contains('http', case=False) & ~train['URL'].str.contains('http') & (train['label']==0)]



# http가 중간에 있으면 악성일까?
train[train['URL'].str.contains('^https', case=False)] # https로 시작해도 악성인게 있는데?




# http만 데이터 확인 (https 말고) : 3649개 -> 정상 439개, 악성 3210개
# 카이제곱 검정 : 관련이 있다 -> 0.0360 정도의 연관이 있다
# P(악성|http)  0.879  >  P(악성|nothttp) 0.223  
# P(정상|http) 0.120  <  P(정상|nothttp) 0.776 
# => http가 있으면 악성일 가능성이 높고, http가 없으면 정상일 가능성이 높음.
A = train['URL'].str.contains(r"http(?!s)", case=False, regex=True)
Derive(train, A, 'http')

# False : 0.999 비율 , True : 0.001 비율  (이 정도 비율은 Cramér's V 값이 안 괜찮게 나오나 봄봄)
a = train['URL'].str.contains(r"http(?!s)", case=False, regex=True).value_counts(normalize=True)
print(f"False : {a[0]:.3f} 비율 , True : {a[1]:.3f} 비율")


# https가 포함됐다고 해서 정상 url인 건 아님.
# https만 데이터 확인 : 8714개 -> 정상 396개, 악성 8318개
# 카이제곱 검정 : 관련이 있다 -> 0.0619 정도의 연관이 있다
# P(악성|https)   0.955  >  P(악성|nothttps)  0.223
# P(정상|https)  0.045  <  P(정상|nothttps)  0.777
# => https가 있으면 악성일 가능성이 높고, https가 없으면 정상일 가능성이 높음.
A = train['URL'].str.contains(r"https", case=False, regex=True)
Derive(train, A, 'https')


# https로 제대로 시작한다고 해서 정상 url인 건 아님.
# https로만 제대로 시작하는 데이터 확인 : 1060개 -> 정상 45개, 악성 1015개
A = train['URL'].str.contains(r"^https", regex=True)
train[train['URL'].str.contains(r"^https", regex=True)]
Derive(train, A, 'https')


# 프로토콜이 ftp인건 없음
train_parsed[train_parsed['URL'].str.contains('^ftp', case=False)]
train_parsed[train_parsed['URL'].str.contains('^ftp://', case=False)]



# -----------------------
# urlparse 이용

url = 'www.jebnt.com/images/index.htm?us.battle.net/login/en/?ref=http\%3A\%2F\%2Fmxvphbpus.battle.net\%2Fd3\%2Fen\%2Findex&amp;app=com-d3'
url = 'fzjyefz.com/images/index.htm?us.battle.net/login/en/?ref=http\%3A\%2F\%2Fus.battle.net\%2Fd3\%2Fen\%2Findex&amp;app=com-d3'

parsed = urlparse(url)
print("scheme:", parsed.scheme)       # https
print("netloc:", parsed.netloc)       # example.com:8080
print("path:", parsed.path)           # /path/page.html
print("query:", parsed.query)         # key=value
print("fragment:", parsed.fragment)   # section




# port에 해당하는 데이터 3140개 -> 정상 393개, 악성 2747개
# 카이제곱 검정 : 관련 있다 -> 0.0331 정도의 연관이 있다
# P(악성|port)   0.875  >  P(악성|notport)  0.223
# P(정상|port)  0.125  <  P(정상|notport)  0.777
train_parsed[train_parsed['port']!='']
A = train_parsed['port']!=''
Derive(train_parsed, A, 'port')

train_parsed['port'].dtypes


url = '%hxxps://my.xero.com/!xkcd/dashboard/'
'://' in '%hxxps://my.xero.com/!xkcd/dashboard/'
safe_urlparse('https://178.156.202.170/sean/mail.php?main_domain=hxxp://emailvs.com&email=kenny@tip.emailvs.com&subdomain=hxxp://emailvs.com')
safe_urlparse('%hxxps://my.xero.com/!xkcd/dashboard/')
safe_urlparse('news:alt.discuss.clubs.public.html.help.gnubee')
safe_urlparse('[email protected].')
safe_urlparse('https://[email protected].')


url = 'https://[email protected].'
parsed = urlparse(url)  # 에러남.

url = '[email protected].'
url = 'yahoo%2eco%[email protected]/'
url = 'https://1992099439:8002/wellsfargo/'
url = complete_url('wsu.edu:8080/~wsherb/images/fabaceae/lupinus.html')
url = complete_url('me@createkindlebooks.org:noobasshole@createkindlebooks.org')
parsed = urlparse(url) # 에러 안 남.
print(parsed.scheme)
print(parsed.netloc)
print(parsed.username)
print(parsed.password)
print(parsed.hostname)
print(parsed.port)
print(parsed.path)
print(parsed.query)
print(parsed.fragment)
safe_urlparse(url)






complete_url('43.159.39.69/v3/signin/identifier?dsh=s-1526075249:1679484868908718&amp;followup=hxxps://accounts.google.com&amp;ifkv=awnoghem6jxsfi3i9gtbq2ptjqew0k8g55g-qujtxuhcxrni5dbry5bb-b5zi9pdc1ibj5ej9nex&amp;passive=1209600&amp;continue=hxxps://accounts.google.com/&amp;flowname=glifwebsignin&amp;flowentry=servicelogin&amp;ifkv=aqmjq7su7s-rvpt4j538kjhtdhbfbyvqzjjwxmh-d2kjeljrqkcajgb1mksnf7hle_5bniwv8xeloa')
complete_url('mss-service.solutions/a1/zohoverify/mail.php?main_domain=hxxp://www.&email=&subdomain=hxxp://www.')

complete_url('%hxxps://my.xero.com/!xkcd/dashboard/')
complete_url('grpvrtual-actv.waw.pl:443/mua/user/scis/j6unvhzsitlyrxstpnfun4tssjgejkn7dldp6fxsjfxo/3d/no-back-button/')
complete_url('www.365882.cc:8443/#/?sharename=365882.cc')
complete_url('www.login.verifyyahoomail.com/login.php?hxxps')
complete_url('inova-bd.com/wp-admin/cgs/b177eabb1d33c4767ea4ad6faa08cf13?hxxps://allegro.pl/auth/oauth/authorize?client_id=tb5sff3crxeyspdn&redirect_uri=hxxps://allegro.pl/login/auth&response_type=code&state=bxmolp&oauth=true')
complete_url('news:alt.discuss.clubs.public.html.help.gnubee')

url = 'inova-bd.com/wp-admin/cgs/b177eabb1d33c4767ea4ad6faa08cf13?hxxps://allegro.pl/auth/oauth/authorize?client_id=tb5sff3crxeyspdn&redirect_uri=hxxps://allegro.pl/login/auth&response_type=code&state=bxmolp&oauth=true'
'/' in url.split('://')[0]

complete_url('[email protected].')
complete_url('[email protected]/')
complete_url('[email protected]..')
complete_url('[email protected]/aol5/a000l.html')
complete_url('yahoo%2eco%[email protected]/')
complete_url('%[email protected]//images/error-log//?i=i&[email protected]')
complete_url('[email protected]/5zk22l6m')
complete_url('[email protected]')
complete_url('[email protected]/go/')
complete_url('ynfp.jp/wp-content/china/[email protected]')

complete_url('1992099439:8002/wellsfargo/')


train_parsed[train_parsed['complete_URL']=='https://178.156.202.170/sean/mail.php?main_domain=hxxp://emailvs.com&email=kenny@tip.emailvs.com&subdomain=hxxp://emailvs.com']



len(train_parsed['count_sc'].unique())  # 242가지의 유니크 가짐 (특수문자 0~673개)
np.sort(train_parsed['count_sc'].unique())

# ['https', 'http', 'news', 'exception', nan, '%hxxps', 'hxxps', '472a4hxxp', 'tel', 'mailto']
train_parsed['scheme'].unique()
train_parsed[train_parsed['scheme']=='https']
train_parsed[train_parsed['scheme']=='http']
train_parsed[train_parsed['scheme']=='news']
train_parsed[train_parsed['scheme']=='%hxxps']
train_parsed[train_parsed['scheme']=='hxxps']
train_parsed[train_parsed['scheme']=='472a4hxxp']
train_parsed[train_parsed['scheme']=='tel']
train_parsed[train_parsed['scheme']=='mailto']
train_parsed[train_parsed['scheme']== 'exception'] # exception이면 [email protected]가 포함된 경우임 (근데 [email protected]가 포함됐다고 무조건 exception인건 아님)
train_parsed[train_parsed['URL'].str.contains('protected')]
train_parsed[train_parsed['scheme'].isna()]


train_parsed['netloc'].unique()

train_parsed['username'].unique()
train_parsed[train_parsed['username']=='get-messenger-premium-features-free']
train_parsed[train_parsed['username']=='direwolf-8c849959a2.herokuapp.com']
train_parsed[train_parsed['username']=='blue-verified-facebook-free']
train_parsed[train_parsed['username']=='adil']
train_parsed[train_parsed['username']=='exu0wgk0298bi9oj8c3f']
train_parsed[train_parsed['username']=='direwolf-1649048faf.herokuapp.com']
train_parsed[train_parsed['username']=='direwolf-50682b0c98.herokuapp.com']
train_parsed[train_parsed['username']=='direwolf-5852e36046.herokuapp.com']
train_parsed[train_parsed['username']=='cassia.martins']
train_parsed[train_parsed['username']=='still-bastion-25750.herokuapp.com']
train_parsed[train_parsed['username']=='fluxmemory.com']
train_parsed[train_parsed['username']=='me@createkindlebooks.org']



train_parsed['password'].unique()

train_parsed['hostname'].unique()

train_parsed['port'].unique()
train_parsed[train_parsed['port']==8080.0]


train_parsed['path'].unique()

train_parsed['query'].unique()
train_parsed[train_parsed['query']=='email=itm-uk@dbk-group.com']

train_parsed['fragment'].unique()

len(train_parsed['count_path'].unique() ) # 133가지
np.sort(train_parsed['count_path'].unique() ) # 0~136 자리수

len(train_parsed['len_query'].unique()) # 770가지
np.sort(train_parsed['len_query'].unique()) # 0~ 8367 자리수


train_parsed.columns






train_parsed['port'].unique()
train_parsed[train_parsed['port']=='8000']

train_parsed['password'].unique()

train_parsed['username'].unique()


# -------------------------------------------------
# path 경로 수
train_parsed['count_path'].unique()

# [시각화] 경로 수 (/ 수)가 많을 수록 정상보다 악성이 많아짐
group_M = train_parsed.groupby('count_path', as_index=False).agg(악성_수=('label','sum'))
group_M1 = train_parsed.groupby('count_path', as_index=False).agg(전체=('label','count'))
group_M = pd.merge(group_M, group_M1, how='left', on='count_path')
group_M['악성_비율'] = group_M['악성_수']/group_M['전체']


plt.figure(figsize=(15,5))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('경로 개수별 악성URL 비율', fontsize=12, pad=20)
plt.xlabel('경로 개수', fontsize=12)  # x축 제목과 글씨 크기 지정
plt.ylabel('악성URL 비율', fontsize=12)  # y축 제목과 글씨 크기 지정
sns.barplot(data=group_M, x='count_path', y='악성_비율')


#
plt.figure(figsize=(10,8))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('경로 개수별 악성.정상 개수', fontsize=12, pad=20)
plt.xlabel('경로 개수', fontsize=12)  
plt.ylabel('악성.정상 개수', fontsize=12)  
sns.countplot(data=train_parsed[train_parsed['count_path']<=10], x='count_path', hue='label')

#
plt.figure(figsize=(10,8))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('경로 개수별 악성.정상 개수', fontsize=12, pad=20)
plt.xlabel('경로 개수', fontsize=12)  
plt.ylabel('악성.정상 개수', fontsize=12)  
sns.countplot(data=train_parsed[train_parsed['count_path']>10], x='count_path', hue='label')

