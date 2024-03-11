################################################
# KNN
################################################
# STEPS:
# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################################
# 1. Exploratory Data Analysis
################################################
df = pd.read_csv("section_datasets/machine_learning/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()
################################################
# 2. Data Preprocessing & Feature Engineering
################################################
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)
random_user = X.sample(1, random_state=45)
knn_model.predict(random_user) #array([1], dtype=int64)
################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob(1 sınıfına ait olma olasılıkları):
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74
# AUC
roc_auc_score(y, y_prob)
# 0.90

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() # 0.73
cv_results['test_f1'].mean() # 0.59
cv_results['test_roc_auc'].mean() # 0.78
# CV methodu daha güvenilir. Değerlerin ilkine göre düşük çıkmış olması,
# tüm dataya validasyon işleminde yanlılık olduğunun sonucudur diye düşünebiliriz. O nedenle CV daha güvenilirdir.

# Başarı Scorelerı nasıl arttırılır??
# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################
knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)
# Fitting 5 folds for each of 48 candidates, totalling 240 fits
# n_jobs=-1 => işlemcinin olası en yüksek performası ile çalışmasını sağlamak içindir.

knn_gs_best.best_params_ # {'n_neighbors': 17}
################################################
# 6. Final Model
################################################
# Birden fazla parametre verilmesinin pratik yolu:
# a=  my_func(**kwargs) => kwargs; dictionary'dir ve içindeki (her) key = variable, (her) value = assigned value to key !!!

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() #0.7669892199303965
cv_results['test_f1'].mean() # 0.6170909049720137
cv_results['test_roc_auc'].mean() #0.8127938504542278

random_user = X.sample(1)
knn_final.predict(random_user) # array([0], dtype=int64)











