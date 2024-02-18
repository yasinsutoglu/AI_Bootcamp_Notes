############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# 1. Veri Ön İşleme
############################################

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel("section_datasets/recommender/datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
# pip install openpyxl
# df_ = pd.read_excel("datasets/online_retail_II.xlsx",
#                     sheet_name="Year 2010-2011", engine="openpyxl")


df.describe().T
df.isnull().sum()
df.shape

# Veri ön işleme fonksiyon kabası
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)

# aykırı değer temizliği için alt/üst limit değerleri belirleme ile veri ön işlemeye devamke
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1 # değişken değerlerinin değişme aralığı
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

#  alt/üst limit değerleri ile baskılama yapmak(outlier'ları kırpmak)
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Veri ön işleme fonksiyon final hali => Quantity ve Price değişkenleri duruma göre değişebilir
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T


############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

df.head()

# Nihai Hale Getirmek İstediğimiz Dataset Formatı:
# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

# Invoice'lar bizim için alışveriş_sepeti/transaction anlamına gelmektedir.
# Sütun değerleri bu sepette olup olmama durumlarıdır.

df_fr = df[df['Country'] == "France"]
# ülke bazında işe başladık
# Örneğin Almanya pazarına girmek istiyorum ve Fransa profil olarak benzer bir ülke.
# Fransa'dan öğrendiğim birliktelik kuralından yola çıkarak Almanya'da tavsiye uygulamaları
# ile pazara girerim.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)
#                                            Quantity
#Invoice Description
#536370   SET 2 TEA TOWELS I LOVE LONDON      24.00000
#        ALARM CLOCK BAKELIKE GREEN           12.00000
#        ALARM CLOCK BAKELIKE PINK            24.00000
#        ALARM CLOCK BAKELIKE RED             24.00000
#        CHARLOTTE BAG DOLLY GIRL DESIGN      20.00000
#                                               ...

# Description'ları sütun olarak düzenlemek için unstack() ya da pivot() fonksiyonu kullanılır.
# iloc[0:5, 0:5] => index based secimle satır ve sütundan 5er tane getir.
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# applymap() => apply fonksiyonu benzeridir apply satır gezerken applymap tüm gözlemleri gezer.
df_fr.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).unstack().fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# İşlemleri Fonksiyonlaştıralım
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_fr, 10120)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################

# min_support=0.01 =>  support threshold değeri
frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

#birliktelik kuralını çıkardığımız aşama:
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)
#leverage değeri kaldırac demek (lift gibidir) ve supportu yüksek olanlara öncelik verme eğilimi vardır.
# Ama lift daha değerli kullanılır bir metric'tir.
# conviction => y ürünü olmadan, x ürününün beklenen değeridir/frekansıdır.
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]
# antecedents => ilk ürünler; consequents => ikinci ürünler

check_id(df_fr, 21086)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)
#confidence'ı yüksek olanları alım için önermeliyiz.

############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
# kişilerin aluşverişi öncesinde bu çalışma yapılmış olur ve hangi ürün alımına
# hangi ürün önerileri olacağı SQL tablosunda tutuluyor olur.
# Bu yordam başka sektörler için de olası senaryolara karsı izlenir.
# Önce olası senaryolara karşı çalışma yapılıp tablo hazırlanır sonra kullanıma açılır.
############################################

# Örnek:
# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id) # ['MINI PAINT SET VINTAGE ']

sorted_rules = rules.sort_values("lift", ascending=False)
# sıralama sonrası antecedents'teki alınan ürüne karşılık consquents'teki ilk ürün tavsiye edilir.

recommendation_list = []
# antecedents içinde tektek gezeceğim ve buradaki ürün/ürünSetlerinde alınan ürünID'sini yakalayıp bu ürün/ürünSetinin bulunduğu
# index numarası ile consequents'taki aynı index'te yer alan ürünSetinden ilkini öneri olarak çıkaracağım.
# list() => listeye çevirme fonksiyonu
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]
# recommendaiton_list içindeki ürünler 0 indexten artan indexlere doğru
# ilerledikçe alım için tavsiye olasılığı daha düşük ürünler olduğu anlamındadır.

check_id(df, 22326) #['ROUND SNACK BOXES SET OF4 WOODLAND ']
# 'MINI PAINT SET VINTAGE ' alınan ürününe karşılık => 'ROUND SNACK BOXES SET OF4 WOODLAND ' tavsiye edilir.

# Tüm İşlemlerimizi Fonksiyonlaştıralım:
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)





