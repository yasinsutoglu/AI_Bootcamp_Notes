###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM -Recency, Frequency, Monetary-)
###############################################################
# RFM Analizi; müşteri segmentasyonu için kullanılan bi tekniktir. Müşterilerin satın alma alışkanlıkları üzerinden
#gruplara ayrılması ve bu gruplar özelinde stratejiler geliştirilebilmesini sağlar.

# RFM Metrikleri: Recency (Yenilik- Müşteri bizden en son ne zaman alışveriş yaptı? Recency : 1 better than 10),
# Frequency (Sıklık - müşterinin yaptığı toplam alışveriş sayısı), Büyük değer olması daha iyidir.
# (Transaction da denir)
# Monetary (Parasal Değer: Müşterinin bize bıraktığı para),  Büyük değer olması daha iyidir.

# Ornek: R=1, F=3, M= 4 ise RFM skoru string concat'tır ve RFM Score = 134 olur.

# RFM Metrikleri, RFM skorlarına çevrilmelidir. Yani tüm metrikleri aynı cinsten ifade etmek demektir.
# Bir nevi standartlaştırma işlemi yapacağız.

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 7. Tüm Sürecin Fonksiyonlaştırılması

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#'%.3f' => virgülden sonra görülmek istenen basamak sayısı

df_ = pd.read_excel("/Users/yasin/Desktop/DSMLBC4/datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape
df.isnull().sum() # hangi değişkenden kaç eksik değer varın cevabıdır.

# essiz urun sayisi nedir? # nunique => (number unique)
df["Description"].nunique()

# value_counts() => aynı satır içeriğinden(index Value) toplam kaç tane olduğu idi.
df["Description"].value_counts().head()

df.groupby("Description").agg({"Quantity": "sum"}).head()

df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

df["Invoice"].nunique() #kaç farklı fatura kesilmişin cevabıdır.

df["TotalPrice"] = df["Quantity"] * df["Price"]

# Her bir faturanın toplam tutarını bulalım.
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()

###############################################################
# 3. Veri Hazırlama (Data Preparation)
###############################################################

df.shape
df.isnull().sum()
df.describe().T
df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True) # dropna() => eksik değerleri silmek için kullanılan fonksiyon.
# inplace=True => değişikliğin kalıcı olmasını sağlar.
df = df[~df["Invoice"].str.contains("C", na=False)]
# ~ => "-nın dışındakiler" anlamındadır

###############################################################
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

# Recency (Analiz Tarihi - Müşteri Son Satınalma Tarihi), Frequency (Müşteri Toplam SAtın Alması),
# Monetary(Müşteri Toplam Harcaması)
df.head()

df["InvoiceDate"].max()

today_date = dt.datetime(2024, 2, 6 )
type(today_date)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
# R : 'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
# F : 'Invoice': lambda Invoice: Invoice.nunique(),
# M : 'TotalPrice': lambda TotalPrice: TotalPrice.sum()
rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]
rfm.shape


###############################################################
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
###############################################################

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
# qcut() nedir? => qcut(VerilenDeğişken, KacEsParcayaBolunecek, BolumSonrasıEtiketler)

# Ornek: 0-100 bolumleme =>  0-20 : (5), 20-40 : (4), 40-60 : (3), 60-80: (2), 80-100: (1)

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
# Quantiles içlerine düşen değerler aynı olunca hata alırsak .rank(method="first")'u kullanırız.
# rank(method="first") => ilk gördüğünü ilk sınıfa atama işine yarar

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

rfm[rfm["RFM_SCORE"] == "55"] # champion sınıf (promosyon falan verilecek :) )

rfm[rfm["RFM_SCORE"] == "11"] # hibernating sınıf

###############################################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
###############################################################

# regex kullanımı start

# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
# regex kullanımı end

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

rfm[rfm["segment"] == "cant_loose"].head() #bu segmentteki müşterileri getirdik (customerID'lerle)
rfm[rfm["segment"] == "cant_loose"].index # CustomerID'lere karşılık gelen index bilgileri

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

new_df["new_customer_id"] = new_df["new_customer_id"].astype(int) # astype(int) => string olarak 12351.00'ı => 12351 yapar

new_df.to_csv("new_customers.csv") #excel'e çevirdik dataframe'i
# csv yerine sql tablosu haline de getirilebilir. Bu sql tablosunu tableu, powerBI yordamıyla
# daha şık görselle raporlama imkanı olur.

rfm.to_csv("rfm.csv")

###############################################################
# 7. Tüm Sürecin Fonksiyonlaştırılması (Tüm sürecin bir scripte çevrilmesi de denebilir)
###############################################################

def create_rfm(dataframe, csv=False):

    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))


    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

################## USAGE OF RFM_FUNC #####################
df = df_.copy()

rfm_new = create_rfm(df, csv=True)

# NIHAI NOTLAR:
# 1) Genel fonksiyonumuzun içindeki alt işlem grupları için ayrı ayrı fonksiyon yazılabilir.
# Bu da program akışında ara katmanlara daha müdahale edilebilir bir yapı eldesi sağlar.
# 2) Bu RFM analiz işlemi dönemsel döngüde tekrar çalıştırılması önem arz edebilir. Örn, her ay
# çalıştırdıktan sonra oluşan segmentleri raporlayabiliriz ama bunu takibi sektörde önemli bir yer tutar.
# Bu takip için bu donemsel csv dosyaları saklanıp karşılaştırmalar yapılarak rapora dökülür.











