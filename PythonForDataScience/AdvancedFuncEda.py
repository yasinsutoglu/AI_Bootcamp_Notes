########################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
# Amaç: Elimize büyük veya küçük boyutlu veri geldiğinde bu veriyi fonksiyonel tarza işleyebilmeyi, hızlı bir şekilde veri ile ilgili iç
# görüler elde edebilmeyi amaçlarız. Yani, hızlı bir şekilde genel fonksiyonlar ile elimize gelen veriyi analiz etmektir.
#########################
# 1. Genel Resim : Veri setinin iç ve dış özellikleri genel hatlarını anlamak ile başlarız. Kaç gözlem var, kaç değişken var,
# ilk gözlem incelemesi, değişken tipleri incelenmesi, eksik değer var mı varsa hangi değişkende kaçar tane var gibi bilgileri edinim
# safhasıdır.
#######################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
# HATIRLATMA : df.tail(), df.shape, df.info(), df.columns, df.index, df.dtypes ,df.describe().T,
# df.isnull().values.any(), df.isnull().sum()

def detail_of_dframe(dataframe, queue=5):
	print("##########-Shape-##########")
	print(dataframe.shape)
	print("##########-Types-##########")
	print(dataframe.dtypes)
	print("##########-Head-##########")
	print(dataframe.head(queue))
	print("##########-Tail-##########")
	print(dataframe.tail(queue))
	print("##########-NA-##########")
	print(dataframe.isnull().sum())
	print("##########-Quantiles-##########")
	print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


detail_of_dframe(df)

# Example-1
df2 = sns.load_dataset("tips")
detail_of_dframe(df2)
# Example-2
df3 = sns.load_dataset("flights")
detail_of_dframe(df3)

##############################
# 2. Analysis of Categorical(Bool-Object-Category) Variables
##############################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

#tek bir değişken analiz etmek istediğimizde:
df["embarked"].value_counts()
df["sex"].unique()  # unique değişkenleri gösterir
df["sex"].nunique() # non-unique sayısı

# pure categoric olanlar
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

# categoric sayılabilecekler : belli bir örneklem sayısı altında olan numerik değişkenleri
# kategorik değişken kapsamına alabiliriz.
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in [
	"float64", "int64"]]

# categoric variable'ın sınıf sayısı cok yüksek ise cardinal variable olarak sayılır.
# Ölçeklenemeyecek kadar fazla sınıfı vardır anlamındadır.
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in [
	"category", "object"]]

# tüm categoric değişkenler nihai olarak:
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique() # kategorik olanlar

[col for col in df.columns if col not in cat_cols] # pure numerik olanlar =>  ['age', 'fare']

# Genelleneyecek Fonksiyonumuzu yazalım!!!
def kat_ozet(dframe, colName, plot=False):
	print(pd.DataFrame({colName: dframe[colName].value_counts(),
	                   "Ratio": 100 * dframe[colName].value_counts() / len(dframe)}))
	print("######################")
	if plot:
		sns.countplot(x=dframe[colName], data = dframe)
		plt.show(block=True)

# bir değişkenin kategorik özetini alma
kat_ozet(df, "sex" , plot=True)

# dataframedeki tüm değişkenlerin döngü ile kategorik özetini alma
for col in cat_cols:
	if df[col].dtype == "bool":
		df[col] = df[col].astype(int)
		kat_ozet(df, col, plot=True)
	else:
		kat_ozet(df, col, plot=True)

# Değişken özelinde Boolean'ı integer(1,0)a çevirdik
df["adult_male"].astype(int)

#################################
# 3. Analysis of Numerical Variables
##################################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

# Kategorik Değişkenleri çıkarımı hatırlayalım
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in [
	"float64", "int64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in [
	"category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

# Numerik değşkenleri çıkaralım şimdi
num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"] ]
# cıktı : ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']

num_cols = [ col for col in num_cols if col not in cat_cols] # ['age', 'fare']

#stats
df[["age", "fare"]].describe().T


# Genelleneyecek Fonksiyonumuzu yazalım!!!
def num_ozet(dframe, numCol, plot=False):
	quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
	print(dframe[numCol].describe(quantiles).T)

	if plot:
		dframe[numCol].hist()
		plt.xlabel(numCol)
		plt.title(numCol)
		plt.show(block=True)


num_ozet(df, "age")

for col in num_cols:
	num_ozet(df, col, plot=True)

############################
# Capturing Variables and Generalizing Operations
############################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# Generalizing Function: (DocString ile)
def gen_col_names(dframe, cat_th=10, car_th=20):
	"""
	Amac: Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini
	verir.
	-----------
	Parameters/Arguments
	----------
	dframe : dataframe
		değişken isimleri alınmak istenen dataframe'dir.
	cat_th : kategorik eşik (int, float)
		numerik fakat kategorik olan değişkenler için sınıf eşik değeri
	car_th : kardinal eşik (int, float)
		kategorik fakat kardinal olan değişkenler için sınıf eşik değeri
	-------
	Returns
	-------
	cat_cols: list
		Kategorik değişken listesi
	num_cols: list
		Numerik değişken listesi
	cat_but_car: list
		Kategorik görünümlü kardinal değişken listesi
	-------
	Notes
	-------
	cat_cols + num_cols + cat_but_car = toplam değişken sayısı
	num_but_cat; cat_cols'un içindedir.
	Return olan 3 liste toplamı toplam değişken sayısına eşittir.
	"""

	# Kategorik Değişkenleri çıkarımı hatırlayalım
	cat_cols = [col for col in dframe.columns if str(dframe[col].dtypes) in
	            ["category", "object", "bool"]]

	num_but_cat = [col for col in dframe.columns if dframe[col].nunique() < cat_th and dframe[
		col].dtypes in ["float64", "int64"]]

	cat_but_car = [col for col in dframe.columns if dframe[col].nunique() > car_th and str(dframe[col].dtypes)
	               in ["category", "object"]]
	cat_cols = cat_cols + num_but_cat
	cat_cols = [col for col in cat_cols if col not in cat_but_car]

	# Numerik değşkenleri çıkaralım şimdi
	num_cols = [col for col in dframe.columns if dframe[col].dtypes in ["int64", "float64"]]
	num_cols = [col for col in num_cols if col not in cat_cols]

	print(f"Observations: {dframe.shape[0]}")
	print(f"Variables: {dframe.shape[1]}")
	print(f"cat_cols: {len(cat_cols)}")
	print(f"num_cols: {len(num_cols)}")
	print(f"cat_but_car: {len(cat_but_car)}")
	print(f"num_but_cat: {len(num_but_cat)}")

	return cat_cols, num_cols, cat_but_car

#########
help(gen_col_names) # docString cağrılır.

catColumns, numColumns, categoricButCardinal = gen_col_names(df)

##########################
# 4. Analysis of Target Variables
##########################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

for col in df.columns:
	if df[col].dtypes == "bool":
		df[col] = df[col].astype(int)

catColumns, numColumns, categoricButCardinal = gen_col_names(df)

# hedef değişken : "survived" => kategorik ve sayısal değişkenler açısından analiz etmek
df["survived"].value_counts()
kat_ozet(df, "survived")

# A. Hedef Değişkenin Kategorik Değişkenler ile Analizi
df.groupby("sex")["survived"].mean()
#sex
#female    0.742038
#male      0.188908

def tar_summary_wCat(dframe, target_col, cat_col):
	print(pd.DataFrame({"Target_Mean": dframe.groupby(cat_col)[target_col].mean()}))


tar_summary_wCat(df, "survived", "sex")
#sex
#female    0.742038
#male      0.188908

for col in cat_cols:
	tar_summary_wCat(df, "survived", col)

# B. Hedef Değişkenin Sayısal Değişkenler ile Analizi
df.groupby("survived")["age"].mean()
#survived
#0    30.626179
#1    28.343690

# alternatif kullanım
df.groupby("survived").agg({"age":"mean"})

def tar_summary_wNum(dframe, target_col, numeric_col):
	print(pd.DataFrame(dframe.groupby(target_col).agg({ numeric_col :"mean"})), end="\n\n")


tar_summary_wNum(df, "survived", "age")
#               age
#survived
#0             30.626179
#1             28.343690

for col in num_cols:
	tar_summary_wNum(df, "survived", col)

###########################
# 5. Analysis of Correlation : Elimize gelen veri setinin ısı haritasıyla korelasyonuna(ilişkisel
# bağımlılık) bakmak amacıyla yaptığımız analizdir. Yüksek korelasyonlu olanların shadow
# rakamlarını incelemekteyiz.
# Korelasyon istatistiksel ölçümü => [-1, 1] aralığındadır.
# 1'e ne kadar yakınsa korelasyon o kadar kuvvetlidir. 0 => korelasyon yok demektir.
#- bir değişken artarken diğeride artıyorsa **pozitif korrelasyon**
#- bir değişken artarken diğeri azalıyorsa **negatif korrelasyon**
##########################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 2:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

# correlation hesaplamak için corr() function kullanılır.
corr = df[num_cols].corr()

#ONEMLI NOT:
# Genellikle birbiri ile yüksek korelasyonda olan değişkenlerin beraber çalışmamasını isteriz
# çünkü ikisi de aynı şeyi ifade ediyor. Çoğunlukla bu değişkenlerden birini çalışma dışında
# bırakmak isteriz.

sns.set(rc={"figure.figsize": (10, 10)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# YUKSEK KRELASYONLU DEGISKENLERIN SILINMESI (İhtiyaç Halinde Kullanılır)

cor_matrix = df.corr().abs()

upper_tri_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

drop_list = [col for col in upper_tri_matrix.columns if any(upper_tri_matrix[col] > 0.90) ]

cor_matrix[drop_list] # droplist edilecek kısmı gördük

#şimdi silelim
df.drop(drop_list, axis=1)


# BU İŞLEMLERİ FONKSIYON HALINE GETIRELIM

def high_correlated_cols(dframe, plot=False, corr_th=0.90):
	corr = dframe.corr()
	cor_matrix = corr.abs()
	upper_tri_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
	drop_list = [col for col in upper_tri_matrix.columns if any(upper_tri_matrix[col] > corr_th)]

	if plot:
		import seaborn as sns
		import matplotlib.pyplot as plt
		sns.set(rc={"figure.figsize": (8, 8)})
		sns.heatmap(corr, cmap="RdBu")
		plt.show()

	return drop_list


high_correlated_cols(df, plot=True)
drop_list = high_correlated_cols(df)

df.drop(drop_list, axis=1)
high_correlated_cols(df, plot=True)


## GUZEL ORNEK
df2 = pd.read_csv("datasets/application_train.csv")
len(df2.columns)
df2.head()

dp_list = high_correlated_cols(df2)
len(df2.drop(dp_list, axis=1).columns)
