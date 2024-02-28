#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("section_datasets/feature/datasets/application_train.csv")
    return data

df = load_application_train()
df.head()


def load():
    data = pd.read_csv("section_datasets/feature/datasets/titanic.csv")
    return data


df2 = load()
df2.head()

#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# 1.1 Aykırı Değerleri Yakalama
#############################################

# Grafik Teknikle Aykırı Değerler

sns.boxplot(x=df2["Age"])
plt.show()

###################
# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df2["Age"].quantile(0.25)
q3 = df2["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df2[(df2["Age"] < low) | (df2["Age"] > up)]

df2[(df2["Age"] < low) | (df2["Age"] > up)].index
# Index([33, 54, 96, 116, 280, 456, 493, 630, 672, 745, 851], dtype='int64')

###################
# Aykırı Değer Var mı Yok mu?
###################

df2[(df2["Age"] < low) | (df2["Age"] > up)].any(axis=None)
df2[~((df2["Age"] < low) | (df2["Age"] > up))].shape #aykırı olmayanlar
df2[(df2["Age"] < low)].any(axis=None)

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df2, "Age") # (-6.6875, 64.8125)
outlier_thresholds(df2, "Fare") # (-26.724, 65.6344)

low, up = outlier_thresholds(df2, "Fare")

df2[(df2["Fare"] < low) | (df2["Fare"] > up)].head()

df2[(df2["Fare"] < low) | (df2["Fare"] > up)].index

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df2, "Age")
check_outlier(df2, "Fare")

###################
# grab_col_names => check_outlier(df2, "col_names") buradaki col_names'i otomatikleştirmek amacındayız.
###################

dff = load_application_train()
dff.head() # burada 122 tane değişken var hangisi numerik hangisi değil ayrıştırıp check_outlier() fonksiyona vermek lazım.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik gözüküp fakat kardinal olan değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler(survived, Pclass etc.) de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    #nihai kategorik listesine gelelim:
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols => tipi object olandan farklı olanları aldık
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

#Fonk. kullanımı:
cat_cols, num_cols, cat_but_car = grab_col_names(df2)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df2, col))


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(df, col))

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

# Aykırı Değerleri Erişen Fonksiyon yazdık.
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

# dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] =>  gözlem sayısını verir
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df2, "Age")

grab_outliers(df2, "Age", True)

age_index = grab_outliers(df2, "Age", True)


outlier_thresholds(df2, "Age")
check_outlier(df2, "Age")
grab_outliers(df2, "Age", True)

# NOT:  Birçok ağaç yöntemi aykırı değerlere duyarlı olması sebebiyle aykırı değerler
# tespit edilip dışarıda bırakılmalıdır. Buraya kadarki işlemleri o nedenle yaptık.

#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################

low, up = outlier_thresholds(df2, "Fare")
df2.shape # (891, 12)

# aykırı değerler haricindeki gözlem sayıları
df2[~((df2["Fare"] < low) | (df2["Fare"] > up))].shape #(775, 12)

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df2)

num_cols = [col for col in num_cols if col not in "PassengerId"] # ['Age', 'Fare']

for col in num_cols:
    new_df = remove_outlier(df2, col)

df2.shape[0] - new_df.shape[0] # 116

###################
# Baskılama Yöntemi (re-assignment with thresholds) : Silme yerine baskılama da tercih edilebilir.
# Baskılamak => limit değer dışındaki değerleri silip yerlerine limit değerleri yazmaktır.
###################

low, up = outlier_thresholds(df2, "Fare")

df2[((df2["Fare"] < low) | (df2["Fare"] > up))]["Fare"]
# üstteki kodun alternatifi
df2.loc[((df2["Fare"] < low) | (df2["Fare"] > up)), "Fare"]

df2.loc[(df2["Fare"] > up), "Fare"] = up

df2.loc[(df2["Fare"] < low), "Fare"] = low

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

# Age True
# Fare True

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))
#Age False
#Fare False

###################
# Recap
###################

df = load()
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")

#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################

# 17, 3 => böyle tek başına aykırı olmayıp beraber aykırılık yaratabilecek değişkenleri
# Çok Değişkenli Aykırı Değerler olarak adlandırabiliriz. (17 yaşında olup 3 kere evlenmek mesela)

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape # (53940, 7)

for col in df.columns:
    print(col, check_outlier(df, col))


low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape #(1889, 7)

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape # (2545, 7)

clf = LocalOutlierFactor(n_neighbors=20) #LOF değerleri üreten fonksiyon
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores => negatifleri pozitiflere döndürdük
np.sort(df_scores)[0:10] # -1'ye yakın iyi, -10'a doğru olanlar daha kötü outlier

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-') # eşik değerini buradaki grafikten dirsek yöntemi
# ile belirleyebiliriz. Burada -5 görünüyor.
plt.show()

th = np.sort(df_scores)[3] #  -4.984151747711709 <= grafikten yola çıkarak 3 değerini aldık

df[df_scores < th]

df[df_scores < th].shape #(3, 7) => bireysel değişken bazında binlerce aykırı
# değişken görünürken burada çoklu değişken analizinde 3 taneye düştü.


df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index) # aykırı değerleri sildik
# ÖNEMLİ NOT:
# az sayıda aykırı değer varken baskılasak ok ama yüzlerce aykırı değer varken baskılama kullanırsak çok sayıda
# duplicate veri olusturarak zorla veriyi bozmuş olacağız. Yani Noise oluşturmuş olacağız. Bu durumda outlier belirleme
# threshold değerlerimiz 0.25 & 0.75 belirlemek yerine 0.05 & 0.95 ya da 0.01 & 0.99 şeklinde belirlemek en mantıklısı olacaktır.
# Çünkü kırpılıp yerine baskılanacak gözlem sayısı az olacaktır bu da sorun yaratmaz.
# Ağaç yöntemleri kullanıyorsak aykırı değerlere dokunmamayı tercih etmeliyiz. Çünkü etkileri zaten düşüktür.
# Doğrusal yöntemleri kullanıyorsak aykırı değer temizliği yapılır.

#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################

df = load()
df.head()

# eksik gozlem var mı yok mu sorgusu
# df.isnull().values => true false'lar matrisi halinde gelir
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)] #[183 rows x 12 columns]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False) #eksiklikleri tüm matrise oransal olarak ele aldık

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
# ['Age', 'Cabin', 'Embarked']

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
#          n_miss  ratio
#Cabin        687  77.10
#Age          177  19.87
#Embarked       2   0.22

missing_values_table(df, True)

#############################################
# Eksik Değer Problemini Çözme
#############################################

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################
df.dropna().shape

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################
#eksik değerleri doldurma => mean, median veya istenen herhangi bir değer olabilir.
df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0) # axis => 0: satır , 1: sütun idi.

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)
#cabin ve embarked(kategorik değişkenler) değişkenlerinde hala eksik değer var.Bunun için mode() kullanırız.

# bir kategorik değişken sütununda eksik değerleri doldurma:
# df["Embarked"].mode()[0] => 'S'
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing") # mode değil de istenilen bir ifade ile doldurduk.

# tüm matriste kategorik değişken eksiklerini doldurma:
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# len(x.unique()) <= 10 : eşsiz değer sayısı 10'dan küçük ise kategorik kabul ettik. Yoksa kardinal kabulü yapabilirdik.

###################
# Kategorik Değişken Kırılımında Değer Atama
###################

df.groupby("Sex")["Age"].mean()

df["Age"].mean()

# df.groupby("Sex")["Age"].transform("mean") => cinsiyete göre grupladığı yaşların ortalamalarını ilgili gruplamaya göre doldurdu.
# fillna() =>groupby'dan gelen kırılımı yakalama propertysi var
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma : ML ile tahmin sonucunda değer atama ama daha sonra detay göreceğiz
#############################################
#Eksikliğe sahip değişken bağımlı değişken diğerleri bağımsız değişken diye düşünüp bir modelleme gerçekleştircez
# Modelleme işlemine göre çıkan tahminlerle eksik değerleri tamamlayacağız.
# KNN => uzaklık temelli algoritma oldugundan dolayı bir standartlaşmaya gitmeliyiz.
df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
#standartlaştırma(amaç: kategorikleri de numerik ifade etme) için burada one-hot encoding olayında get_dummies() kullandık.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)  # drop_first=True => iki sınıfa sahip
# kategorik değişkenlerin ilkini at ikincisini tut.
# burada son versiyon için "+" yemedi!!!

dff.head()

# değişkenlerin standartlatırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması.
from sklearn.impute import KNNImputer
# KNNImputer => en yakın değerli komşulara bakar, komşulardaki değerlerin ortalamasını alıp eksik yere atar.
# Uzaklık prensibine göre çalışır.

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# daha önce yapılan standartlaştırma işlemini geri alıp gerçek değerleri görmek: inverse_transform()
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]


###################
# Recap
###################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma

#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

msno.bar(df) # bar() => değişkenlerin NaN olmayan tam sayılarını bar halinde verir
plt.show()

msno.matrix(df) # matrix() => değişken eksikliklerin birlikte çıkıp/çıkmadığı ile ilgili bilgi edinmek için
plt.show()

msno.heatmap(df) # heatmap() => nullity correlation değerlerini suna bize
plt.show()
# +1'ye yakın değerlerde; eksiklik bağımlı iki değişkende eş zamanlı ortaya çıktığı düşünülür.
# -1'ye yakın değerlerde; eksiklik bağımlı birinde varken diğerinde yok şeklinde tezahür ettiği düşünülür.

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################
missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns # listComprehension ile de yapılabilirdi.

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)

###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)


#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# Algoritmaların bizden beklediği veriye uygun/standart hale getirmek ilk amacımız.
# İkinci amacımız ise model tahmin yöntemlerine iyileştirmeler sağlamaktır.
#############################################

#############################################
# Label Encoding & Binary Encoding
#############################################

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder() # fonksiyo ile nesne olarak tanımlama
le.fit_transform(df["Sex"])[0:5] # array([1, 0, 0, 0, 1]) => alfabatik sıraya göre ilk olan(female) 0'dır.
le.inverse_transform([0, 1]) # array(['female', 'male'], dtype=object) => dönüştürme bilgilerini hatırlamak için kullanabiliriz.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

# Elimde yüzlerce değişken varsa binary columns (kategorik değişkenler) seçmek için kullanırız bunu:
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head() # dönüştürme sonrası biz 0 ve 1'ler görmek isteriz ama
# 2 de gözlemlersek bunlar eksik değerleri temsil eder.Bunu unutmamak lazım!


for col in binary_cols:
    label_encoder(df, col)


df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())
# nunique() & unique() ayrımının farkında olmak önemli!! unique() => NaN'ları da dahil eder.

#############################################
# One-Hot Encoding
#############################################
df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head() #dummy_na=True =>  eksik değerler için de sınıf olusturur.

pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head() # burada hem binary encode , hem one-hot encode yapabilmiş oluyoruz.

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df) => alttaki code için alternatif
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()
df.head()

#############################################
# BONUS BÖLÜM: Rare Encoding
#############################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.
###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# cat_summary => kategorik değişkenlerinin sınıflarını ve oranlarını gösterir
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# kategorik değişkenler altında frekansı düşük gereksiz sınıfları gözlemledik onları çıkarmalı
# veya rare encoding ile yola devam etmeliyiz.

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################
df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# target => bağımlı değişken, cat_cols => kategorik değişken
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts())) # kaç sınıf var bilgisi
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)
# Aşağıdakini örnek alarak konuşacaksak mesela %1 altındakileri Rare olarak toplayabiliriz.
# DEF_60_CNT_SOCIAL_CIRCLE : 9
#                            COUNT     RATIO  TARGET_MEAN
# DEF_60_CNT_SOCIAL_CIRCLE
# 0.0                       280721  0.912881     0.078348
# 1.0                        21841  0.071025     0.105169
# 2.0                         3170  0.010309     0.121451
# Alttakiler RARE olabilir.
# 3.0                          598  0.001945     0.158863
# 4.0                          135  0.000439     0.111111
# 5.0                           20  0.000065     0.150000
# 6.0                            3  0.000010     0.000000
# 7.0                            1  0.000003     0.000000
# 24.0                           1  0.000003     0.000000

#############################################
# 3. Rare encoder'ın yazılması ile kalıcı olarak veri seti güncellenmesi.
#############################################
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

# 0.01 oranı altında kalan kategorik değişken sınıflarını bir araya getirecek.
new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()

#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################
###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
# eksik ve aykırı değerlerden etkilenme söz konusu burada yine.
###################

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

###################
# RobustScaler: Medyanı çıkar IQR'a böl.
# StandardScaler'e göre daha güvenilir (aykırı değerlere karşı etkilenmeme). Ama çok yaygın kullanılmaz piyasada nedense.
###################
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
# yaygın kullanılır. Özel bir aralık için çok kullanılır
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]

# sayısal değişkenlerin çeyreklik değerlerini göstermek ve grafik oluşturmak için kullandık
# ana resmi görmek için
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)

#############################################
# FEATURE EXTRACTION (Özellik Çıkarımı)
#############################################
#############################################
# Binary Features: Flag, Bool, True-False => var olan değişkenler üzerinden yeni değişkenler türetmek
#############################################

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')
# Cabin'de dolu olanlara 1 , NaN olanlara 0 vererek yeni değişken oluşturduk
#Cabin   NEW_CABIN_BOOL
#NaN             0
#C85             1
#NaN             0

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})
# Anlamlı kaydadeğer veriye döndü:
#                 Survived
# NEW_CABIN_BOOL
# 0                  0.300
# 1                  0.667


from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 9.4597, p-value = 0.0000
# H0 red, dolayısıyla aralarında anlamlı farklılık var denebilir.

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})
#               Survived
# NEW_IS_ALONE
# NO               0.506
# YES              0.304


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = -6.0704, p-value = 0.0000
# H0 red, dolayısıyla aralarında anlamlı farklılık var denebilir

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

df.head()

###################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})
#                 mean count
# NEW_NAME_DR
# 0              0.383   881
# 1              0.500    10

###################
# Regex ile Değişken Türetmek
###################

df.head()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})
#         Survived   Age
#               mean count   mean
# NEW_TITLE
# Capt         0.000     1 70.000
# Col          0.500     2 58.000
# Countess     1.000     1 33.000
# Don          0.000     1 40.000
# Dr           0.429     6 42.000
# Jonkheer     0.000     1 38.000
# Lady         1.000     1 48.000
# Major        0.500     2 48.500
# Master       0.575    36  4.574
# Miss         0.698   146 21.774
# Mlle         1.000     2 24.000
# Mme          1.000     1 24.000
# Mr           0.157   398 32.368
# Mrs          0.792   108 35.898
# Ms           1.000     1 28.000
# Rev          0.000     6 43.167
# Sir          1.000     1 49.000

#############################################
# Date Değişkenleri Üretmek
#############################################

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()
# date modulu incelenebilir daha detay gerekirse

#############################################
# Feature Interactions (Özellik Etkileşimleri) : Değişkenlerin birbiryle etkileşime girmesi demektir.
#############################################
df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1 # 1 => kişinin kendisi

df.loc[(df['SEX'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['SEX'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['SEX'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['SEX'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.groupby("NEW_SEX_CAT")["Survived"].mean()

#############################################
# UYGULAMA : Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################
# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#############################################
# 4. Label Encoding
#############################################

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

#############################################
# 5. Rare Encoding
#############################################

rare_analyser(df, "SURVIVED", cat_cols)

df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#############################################
# 6. One-Hot Encoding
#############################################
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape
#############################################
# 8. Model
#############################################

y = df["SURVIVED"] # bağımlı değişken
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1) # bağımsız değişkenler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)


