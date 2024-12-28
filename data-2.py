import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일에서 데이터 읽기
df = pd.read_csv('employee_data.csv')
print(df.head())  # 데이터의 처음 5행을 출력

df.to_csv('output.csv', index=False)

# 데이터프레임의 정보 확인
print(df.info())

# 통계 요약 정보
print(df.describe())

# 특정 열 선택
names = df['이름']
print(names.head())

# 특정 행 선택 (인덱스로) 0이 첫번째 행.
first_row = df.iloc[0]

# 조건을 이용한 필터링
older_than_30 = df[df['나이'] > 30]
print(older_than_30.head())





