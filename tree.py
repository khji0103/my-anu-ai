# 1. 필요한 라이브러리 임포트
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 2. 데이터 준비 및 탐색
# 데이터셋 로드
wine = load_wine()
X = wine.data
y = wine.target

# 데이터셋을 DataFrame으로 변환 및 탐색
wine_pd = pd.DataFrame(wine.data, columns=wine.feature_names)

# 데이터 특성 정보 출력
print("\n=== 각 특성의 최소/최대값 ===")
for feature in wine.feature_names:
    min_val = wine_pd[feature].min()
    max_val = wine_pd[feature].max()
    print(f"{feature}: [{min_val:.2f} ~ {max_val:.2f}]")

# 3. 모델 준비 및 학습
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 4. 모델 평가
accuracy = model.score(X_test, y_test)
print(f"\n모델 정확도: {accuracy:.2f}")

# 5. 모델 시각화
# 결정 트리 그래프 시각화
plt.figure(figsize=(10, 8))
tree.plot_tree(model, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.show()

# 텍스트 형태로 트리 출력
tree_rules = export_text(model, feature_names=wine.feature_names)
print("\n=== 결정 트리 규칙 ===")
print(tree_rules)

# 6. 새로운 데이터 예측
# 랜덤 샘플 생성
sample_features = []
random_sample = []

for feature in wine.feature_names:
    min_val = wine_pd[feature].min()
    max_val = wine_pd[feature].max()
    random_value = np.random.uniform(min_val, max_val)
    random_sample.append(random_value)

sample_features.append(random_sample)


# 예측 수행 및 결과 출력
prediction = model.predict(sample_features)

print(f"\n=== 예측 결과 ===")
print(f"모델 정확도: {accuracy:.2f}")  # accuracy 출력 추가
print(f"랜덤하게 생성된 샘플 데이터:")
for feature_name, value in zip(wine.feature_names, sample_features[0]):
    print(f"{feature_name}: {value:.2f}")
print(f"\n예측된 와인 종류: {wine.target_names[prediction[0]]}","입니다")