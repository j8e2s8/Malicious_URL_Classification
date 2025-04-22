import pandas as pd
import numpy as np

# 데이터 불러오기
# train = pd.read_csv('../data/raw/train.csv')
# train.to_parquet("../data/raw/train.parquet", compression="snappy")
# train.sample(10, random_state=42).to_csv("../data/sample_train.csv", index=False)

train = pd.read_parquet('../data/raw/train.parquet')
train.head(5)

test = pd.read_csv('../data/raw/test.csv')
test.head(5)

submission = pd.read_csv('../data/raw/sample_submission.csv')
submission.head(5)


# 데이터 정보
train.shape   # (6995056, 3)
test.shape   # (1747689, 2)
