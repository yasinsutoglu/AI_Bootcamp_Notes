################################
# Unsupervised Learning
################################

# !pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

################################
# A. K-Means
################################
df = pd.read_csv("section_datasets/machine_learning/USArrests.csv", index_col=0)

df.head()
df.isnull().sum()
df.info()
df.describe().T

# uzaklık temelli işlemler yaptığımızdan standardlaştırmaya tabi tuttuk
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters #dışarıdan belirlenmesi gereken ve ayarlanması gereken bir hiperparametredir.
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_ # SSE/SSR/SSD denilen değer.=> 3.6834561535859134

################################
# Optimum Küme Sayısının Belirlenmesi
################################
kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# yukarıdaki cıkacak grafikten kendimiz optimum küme sayısı belirlemek yerine KElbowVisualizer() fonksiyonunu kullanırız.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv("section_datasets/machine_learning/USArrests.csv", index_col=0)

df["cluster"] = clusters_kmeans

df.head()

df["cluster"] = df["cluster"] + 1

df[df["cluster"]==1]

df.groupby("cluster").agg(["count","mean","median"])

df.to_csv("clusters.csv")

################################
# B. Hierarchical Clustering
################################

df = pd.read_csv("section_datasets/machine_learning/USArrests.csv", index_col=0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

# birleştirici clustering yöntemini uygulayalım => öklid uzaklığına göre gözlem birimlerini kümelere ayırıyor.
hc_average = linkage(df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()


plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10, # 10 küme olsun dedik
           show_contracted=True,
           leaf_font_size=10)
plt.show()

################################
# Kume Sayısını Belirlemek
################################
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
clusters = cluster.fit_predict(df)

df = pd.read_csv("section_datasets/machine_learning/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = clusters_kmeans
df["kmeans_cluster_no"] = df["kmeans_cluster_no"]  + 1

################################
# C. Principal Component Analysis
################################

df = pd.read_csv("section_datasets/machine_learning/hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)
df.shape #(322, 16)

# standardlaştırma yaptık
df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_ # bu method başarısı varyans oranlarına göre belirlenmektedir.
np.cumsum(pca.explained_variance_ratio_)
# cumulative variance'lar => toplam ne kadar
# açıklama oranına sahip bunlar verisetinde demektir!

################################
# Optimum Bileşen Sayısı
################################

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show() # grafikten %82'ye karşılık gelen 3 bileşen sayısını seçtik.

################################
# Final PCA'in Oluşturulması
################################

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_ #array([0.46037855, 0.26039849, 0.1033886 ])
np.cumsum(pca.explained_variance_ratio_) # array([0.46037855, 0.72077704, 0.82416565])

################################
# BONUS: Principal Component Regression => bu method'da önce PCA uygulanıyor
# sonra bu bileşenlerin üzerine bir regresyon modeli kuruluyor.
################################
# Varsayım: Bu verisetinde değişkenler arasında çoklu doğrusal bağlantı problemi var.
# Değişkenler arası yüksek korelasyon olunca olur.
df = pd.read_csv("section_datasets/machine_learning/hitters.csv")
df.shape
len(pca_fit)

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols) # 16

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head()

df[others].head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1)
final_df.head()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# get dummies yada one-hot encoder da kullanılabilir.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error"))) #  345.6021106351967
y.mean() # 535.9258821292775 => rmse göre yüksek ama ne iyi ne kötü

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error"))) #401.0279978363749

# HiperParametre optimizasyonu
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}
# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error"))) # 330.1964109339104
# ÖNEMLİ SORU : Elimde bir veri seti var ama verisetinde label yok ama sınıflandırma modeli kurmak istiyorum. Ne yapmalı?
#CVP : Önce unsupervised şekilde clusterlar cıkarırım, çıkan clusterlar = sınıflar(label) diyebilirim etiketleyerek,
# sonra da bunu bir sınıflandırıcıya sokarım. Yeni müşteri geldiğinde onun özelliklerine göre hangi cluster'a ait olduğunu bulabilirim.

################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

################################
# Breast Cancer
################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("section_datasets/machine_learning/breast_cancer.csv")
df.shape # (569, 32)

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)

def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y) # verisetini 2 bileşene indirgedik

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")

################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")

################################
# Diabetes
################################

df = pd.read_csv("section_datasets/machine_learning/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")




















