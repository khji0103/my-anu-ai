import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data.csv')
print(df)

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
monthly_data = df.resample('M').sum()

print(monthly_data)

# 3. 데이터 시각화
# Matplotlib을 사용하여 시각화합니다.
plt.figure(figsize=(10, 6)) #가로 세로 길이
plt.plot(monthly_data.index, monthly_data['value'], marker='o')
plt.title('Monthly Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

#시각화 후 회사 클라우드에 저장까지 해줘야함

