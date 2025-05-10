# 🛰️ Datanauts: 악성 URL 분류 프로젝트

악성 URL을 탐지하기 위한 머신러닝 기반 분류 프로젝트입니다.  
실제 보안 업무에서 발생할 수 있는 URL 기반 위협 요소를 식별하며, 전처리부터 모델 해석, 대시보드 구현까지 실무 흐름을 포트폴리오로 구성하였습니다.

---

## 📌 프로젝트 개요

- 전처리 및 URL 구조 기반 자연어 처리
- TF-IDF 벡터화 및 주요 특성 추출
- 머신러닝 기반 분류 모델 (LGBM, XGBoost 등)
- 모델 해석 (SHAP을 활용한 feature importance 분석)
- Streamlit 기반 시각화 대시보드 구현
- 실무 중심 문서화 및 재현 가능한 코드 구성

---

## 📁 데이터 소개

- **Train 데이터** (6,995,056건):  
  [📥 Google Drive에서 다운로드](https://drive.google.com/file/d/1fPr5oApE65LbWQiRGt8q8mbO37YmZHxV/view?usp=drive_link)

- **Test 데이터** (1,747,689건):  
  `data/raw/test.csv`

- **Train 샘플 데이터** (50,0000건):  
  `data/sample_train.csv`

- **Test 샘플 데이터** (12,5000건):  
  `data/sample_test.csv`

- **Train EDA한 데이터** (6,995,056건):  
  `data/eda_train0.csv`
  `data/eda_train1.csv`
  `data/eda_train3.csv`
  `data/eda_train4.csv`
  `data/eda_train5.csv`
  `data/eda_train6.csv`
  `data/eda_train7.csv`
  `data/eda_train8.csv`
  `data/eda_train9.csv`
  `data/eda_train10.csv`
  `data/eda_train11.csv`
  `data/eda_train12.csv`
  `data/eda_train13.csv`

- **Test EDA한 데이터** (1,747,689건):  
  `data/eda_test0.csv`
  `data/eda_test1.csv`

---

## 🧠 주요 기술 스택

- **언어/환경**: Python, Jupyter Notebook  
- **모델링**: LightGBM, XGBoost, RandomForest  
- **전처리**: TF-IDF, URL 파싱, 특수문자/패턴 추출  
- **해석 및 검증**: SHAP, confusion matrix, threshold optimization  
- **시각화**: Matplotlib, Seaborn, Streamlit
- **협업/형상관리**: Git, GitHub

---

## 📊 결과 요약

- 악성 URL 분류 정확도: **XX.X%**
- 최종 모델: **LightGBM + threshold 조정**
- SHAP 기반 주요 특징:
  - 특수문자 수, 도메인 패턴, 피싱 키워드 포함 여부 등

---

## 🚀 실행 방법

```bash
# 패키지 설치
pip install -r requirements.txt

# 모델 학습 및 결과 생성
python train.py

# 대시보드 실행
streamlit run app/dashboard.py