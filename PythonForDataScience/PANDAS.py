#############################
# PANDAS => Veri analizi deyince akla gelen ilk lib'lerden biridir. Ekonomik ve finansal çalışmalr için oluşturulmuştur.2008de çıktı.
################################
# Pandas Series : En yaygın veri yapılarıdır. Tek boyutlu ve index bilgisi barındıran veri tipidir.
# Pandas DataFrame: Çok boyutlu veri barındıran veri tipidir. Üzerinde çalışılacak ana veri yapısı denebilir.
# Reading Data
# Quick look at data
# Selection in Pandas
# Aggregation & Grouping
# Apply & LAmbda
# Join işlemleri

########################
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5]) # arguman olarak verilen liste'yi pandas serisine çevirir. Index bilgisi bir iç özellik olarak
# görülebilir.
#0    10
#1    77
#2    12
#3     4
#4     5
#dtype: int64

type(s) # pandas.core.series.Series
s.index # RangeIndex(start=0, stop=5, step=1)
s.dtype #int64
s.size #toplam eleman sayısı bilgisi
s.ndim #dimension bilgisi
s.values
type(s.values) # numpy.ndarray

s.head(3) #verilerden ilk üçünü getir demektir => s.head() : ilk 5 tane getirir default
s.tail(4) # verilerden son dördünü getir demektir

######################
# READING DATA :csv, txt, xslx dosya formatlarını okumayı özel pandas fonksiyonları kullanarak yaparız.
#####################
import pandas as pd

df = pd.read_csv("datasets/advertising.csv") # pd üzerine ctrl + sol tık ile gidince "read_" diye aratırsak data okuma ile ilgili
# methodlara ulaşabiliriz. İlgili fonksiyon detayına(doc_string) yine ctrl + sol tık ile gidilir.
df.head()

####################
# Quick look at data (SEABORN kullandık)
####################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic") #Seaborn lib'ten hazır veri setlerine erişim için kullandığımız method.
df.head() #survived değişkeni bağımlı(hedef) değişken 0/1 => false/true(dead/alive)
df.tail()
df.shape # boyut bilgisi almada kullanırız => (891, 15): satır-sutun
df.info() # object ve category tipleri => kategorik değişkendir
df.columns #Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'adult_male', 'deck',
# 'embark_town', 'alive', 'alone'], dtype='object')
df.index # RangeIndex(start=0, stop=891, step=1)
df.describe().T # verinin özet istatistiklerine erişim için kullanılır. T : transpoze al demek.
df.isnull().values.any() # veri setindeki values'da herhangi birinde null var mı?? => true/false
df.isnull().sum() #df.isnull().min() => bu ikisi de df.isnull()'daki true'ları 1 false'ları 0 sayar. Her değişkende kaç tane eksik değer
# bilgisi olduğunu hesapladık. Örnek :deck => 688
df["sex"].head() # bir dataframe(veriSeti)den değişken(columnName) seçmek istersek df["varName"] -OR- df.varName şeklinde kullanım yaparız.
df["sex"].value_counts()
#CIKTI:
#sex
#male      577
#female    314
#Name: count, dtype: int64

####################
# SELECTION IN PANDAS : dataframe'deki satır ve sutundaki verilere erişim
####################
import pandas as pd
import seaborn as sns
df2 = sns.load_dataset("titanic")
df2[0:13] # ilk 13 satır veri gelir
df2.drop(0, axis=0).head() # 0'ncı indeksteki(satır) veriyi sildik.
#  axis : {0 or 'index', 1 or 'columns'}, default 0
#             Whether to drop labels from the index (0 or 'index') or
#             columns (1 or 'columns').

delete_indexes = [1, 3, 5, 7]
df2.drop(delete_indexes, axis=0).head(7)
#df2 = df2.drop(delete_indexes, axis=0)  -OR-  df2.drop(delete_indexes, axis=0, inplace=True)  => Silmeyi kalıcı hale getirdi
# inplace=True => bir değişiklik yapılınca dataframe'de bu değişikliğin kalıcı olması gerektiği bilgisini veren argümandır. Birçok
# method'da yer almaktadır.


##############################
# Eldeki Değişkeni indexe Çevirme
# Örn: Yaş değişkenini(column) index'e (satır numaralandırması) ; indexteki bir değeri değişkene (column'a) almak istersek
############################

df2["age"].head() # df2.age.head()

df2.index = df2.age # Index([22.0, 38.0, 26.0, 35.0, 35.0,  nan, 54.0,  2.0, 27.0, 14.0, ... 33.0, 22.0, 28.0, 25.0, 39.0, 27.0, 19.0,
# nan, 26.0, 32.0], dtype='float64', name='age', length=891)

# Dönüşüm işlemi sonrası bu veriyi sütunlardan silmeliyiz
df2.drop("age", axis = 1).head()
df2.drop("age", axis = 1, inplace=True) # axis = 1 => sütun silme ; axis = 0 => satır sileme

#INDEXI DEGISKENE CEVIRME
df2.index
df2["age"] # run et => 'key' Error
# Eğer "age" değişkeni dataframe'de yoksa bu yeni ekleme olarak algılanır.

df2["yasin"]  = df2.index
df2.head(3) # column names => survived  pclass   sex  sibsp  ...  embark_town  alive  alone yasin
# "yasin column"un altına index'teki değerleri atar.

df2 = df2.reset_index().head()
# df2.reset_index() => index'teki "age" bilgisi gider yine numaralandirma 0,1,2.... olarak yapılır. "age" index
# bilgisi dataframe'e en baş sütun olarak eklenir.


####################
# Değişken (Columns) Üzerinde İşlemler
####################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None) # gösterimde ... ile gösterilmeyen sutun isimleri gösterilmesini setledik.
df3 = sns.load_dataset("titanic")
df3.head()

"age" in df3 #True doner

df3["age"].head()
### VERY IMPORTANT ####
type(df3["age"].head()) #pandas.core.series.Series
type(df3[["age"]].head()) #pandas.core.frame.DataFrame

df3[["age", "alive"]] #bir dataframe'den birden fazla değişken seçmek istersek ilgili columnName'leri df3[]'ün içine liste olarak
# vermeliyiz.
col_names = ["age", "adult_male", "alive"]
df3[col_names]

df3["yasKare"] = df3["age"]**2
df3.head() #yasKare sütunu ve değerleri en son sütun olarak eklendiğini gördük

df3["yeniYas"] = df3["age"] / df3["yasKare"] # bu tarz kullanım da mevcuttur.
df3.head()

# birden fazla sütun silmek istersek:
col_drop = ["yasKare", "yeniYas"]
df3.drop(col_drop, axis=1).head()
df3.drop(col_drop, axis = 1, inplace=True)

df3.loc[:, ~df.columns.str.contains("age")].head() # sütun isminde "age" içermeyen sütun isimlerine göre verileri getir
# : => tüm satırlar oldu
# ~ => değilini almak
# loc => label based order columns : dataframe'lerde seçmek için kullanılan özel yapı

###################
# LOC & ILOC : dataframe'lerde seçim işlemlerinde kullanılan özel yapılardır.
# ILOC : integer based selection
# LOC : label based selection
###################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df4 = sns.load_dataset("titanic")
df4.head()

df4.iloc[0:3] # ilk üç satırı getirir - 0,1,2nci satırlar
df4.iloc[0, 0] # 0ncı satır 0ncı sütun elemanını getir

df4.loc[0:3] # ilk 4 satır geldi - 0,1,2,3ncu satırlar (loc-isimlendirmenin kendisini seçiyor)

df4.iloc[0:3, "age"] # hata verir çünkü sütun için de integer değer bekliyor
df4.iloc[0:3, 0:3]
#CIKTI:
#  survived  pclass     sex
#0         0       3    male
#1         1       1  female
#2         1       3  female

df4.loc[0:3, "age"] # dogru çalışır

# Güzel Örnek
col_names = ["age", "embarked", "alive"]
df4.loc[0:3, col_names]

########################
# CONDITIONAL SELECTION
# df[...CONDITIONAL PHRASE...] şeklinde yazılır => yeni df_new olur. df_new["age"] gibi seçim yapılır.
########################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df5 = sns.load_dataset("titanic")
df5.head()

df5[df5["age"] > 50].head()
df5[df5["age"] > 50]["age"].count() # yaşı 50den büyük olanların sayısını aldık

df5.loc[df5["age"] > 50 , ["age", "class"]].head()
# CIKTI
#     age   class
#6   54.0   First
#11  58.0   First
#15  55.0  Second
#33  66.0  Second
#54  65.0   First

# Birden fazla koşul giriliyorsa koşullar parantez içine alınmalı
df_new = df5.loc[(df5["age"] > 50)
        & (df["sex"] == "male")
        & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton") ),
		["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()
#embark_town
#Southampton    35
#Cherbourg       9

#########################
# AGGREGATION & GROUPING(Kırmak, kırılım da deniyor)
########################
# Toplulaştırmaya ait özel Fonksiyonları şöyle sıralayabiliriz;
# count() , first(), last(), mean(), median(), min(), max(), std(), var(), sum(), pivot table
# Bu fonksiyonlarının öncesinde groupby() ile gruplarız.

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df6 = sns.load_dataset("titanic")
df6.head()


df6["age"].mean() #herkesin yaş ortalaması

df6.groupby("sex")["age"].mean() #cinsiyete göre yaş ortalaması

# DAHA ŞEKILSUKUL KULLANIM
df6.groupby("sex").agg({"age": "mean"}) #cinsiyete göre yaş ortalaması ; {"columnName" : "aggregationFuncName" }
df6.groupby("sex").agg({"age": ["mean", "sum"]})


df6.groupby("sex").agg({"age": ["mean", "sum"],
                         "survived": "mean"})

df6.groupby(["sex", "embark_town"]).agg({"age": ["mean" , "sum"],
                         "survived": "mean"})
##CIKTI:
#                          age            survived
#                         mean       sum      mean
#sex    embark_town
#female Cherbourg    28.344262   1729.00  0.876712
#       Queenstown   24.291667    291.50  0.750000
#       Southampton  27.771505   5165.50  0.689655
#male   Cherbourg    32.998841   2276.92  0.305263
#       Queenstown   30.937500    495.00  0.073171
#       Southampton  30.291440  11147.25  0.174603

df6.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean" , "sum"],
                                                  "survived": "mean",
                                                  "sex": "count"})

########################
# PIVOT TABLE : groupby'a benzer sekilde veri setini kırılımlar acısından degerlendirmek ve ilgilendiğimiz özet istatistiği bu kırılımlar
# açısından görme imkanı sağlar.
# pivot_table("kesişim", "satırdaGorunecek", "sutundaGorunecek") => kesişimlerde görmek istediğim columName'leri girerim.
# pivot_table'ın default değeri mean()'dir. Yani kesişimlerin otomatik ortalaması alınarak gösterillir.
#########################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df7 = sns.load_dataset("titanic")
df7.head()

df7.pivot_table("survived", "sex" , "embarked")

df7.pivot_table("survived", "sex" , "embarked", aggfunc="std")

df7.pivot_table("survived", ["sex"] , ["embarked", "class"]) # satır ve sutunlarda cok seviyeli(branching) gösterim

# cut() ve qcut() =>  eldeki sayısal değişkenleri kategorik değişkenlere çevirmeye yarayan en yaygın fonk'lardır.
# cut(-neyi boleceğim-, -nerelerden böleceğim-) => eliindeki sayısal değişkeni hangi kategorik değişkene bölmek istediğini biliyorsan
# qcut() => eliindeki sayısal değişkeni tanımıyorum çeyreklik değerlere göre bölmek istersem
df7["new_age"] = pd.cut(df7["age"], [0, 10, 18, 25, 40, 90])
df7.pivot_table("survived", "sex", ["new_age", "class"])

########################
# APPLY & LAMBDA
# Apply => satır veya sutunlarda otomatik fonksiyon çalıştırmaya yarar
# Lambda bir  fonksiyon tanımlama şeklidir fakat kullan-at tipidir. JS IIFE gibi yani.
#########################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df8 = sns.load_dataset("titanic")
df8.head()

df8["age2"] = df8["age"] * 2
df8["age3"] = df8["age"] * 5

(df8["age"]/10).head() # tek kolon için işlem yapmak basit ama ya çok kolona uygulamak istersek??

# temel çözüm döngüdür
for col in df8.columns:
	if "age" in col:
		df8[col] = (df8[col] / 10).head()

df8.head()

# apply ile pratik çözüm
df8[["age", "age2", "age3"]].apply(lambda x:x / 10).head()
df8.loc[:, df8.columns.str.contains("age")].apply(lambda x:x / 10).head()
# ONEMLI NOT:  aslında apply() => JS map() gibi ve lambda da JS arrow function gibi
df8.loc[:, df8.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std() ).head() # standartization yaptık

# FARKLI KULLANIM
def standart_scaler(col_name):
	return (col_name - col_name.mean()) / col_name.std()

df8.loc[:, df8.columns.str.contains("age")] = df8.loc[:, df8.columns.str.contains("age")].apply(standart_scaler).head()
# df8.loc[:,["age", "age2", "age3"]] = df8.loc[:, df8.columns.str.contains("age")].apply(standart_scaler).head()
df8.head()

###################
# JOIN ISLEMLERI
# Concat : n(A) + n(B) (disjoint sets)
# Merge : n(A∪B) = n(A) + n(B) − n(A∩B)
###################

# DataFrame(veriYapısı, değişkenAdları ) => sıfırdan dataframe olusturmaya yarar
# veriYapısı => list, dict, numpyArray olabilir ben bunu dataframe'e çevirecem der
# değişkenAdları => columns = [....]

import pandas as pd
import numpy as np
m = np.random.randint(1, 30, size=(5,3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2]) # iki dataframe altalta birleştirme yapıldı
pd.concat([df1, df2], ignore_index=True) # ignore_index=True => indexlerin aynı kalıp karışıklık olusturmasını engeller. Indexler düzgün
# sıralı olur.

#  axis : {0/'index', 1/'columns'}, default 0
#         The axis to concatenate along.

###### MERGE  #########

df3 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                   'group': ['accounting', 'engineering', 'engineering', 'hr']})

df4 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                   'start_date': [2010, 2009, 2014, 2019]})

df5 = pd.merge(df3, df4)
pd.merge(df3, df4, on = "employees")

df6 = pd.DataFrame({ 'group': ['accounting', 'engineering', 'hr'],
                   'manager': ['caner', 'mustafa', 'osman']})

pd.merge(df5, df6)



