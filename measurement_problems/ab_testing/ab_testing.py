######################################################
# Temel İstatistik Kavramları
######################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


############################
# Sampling (Örnekleme) :  Ana kütlenin bir temsilcisidir.
############################

populasyon = np.random.randint(0, 80, 10000) # mahalledeki insanların yaşlarını temsil etsin
populasyon.mean()

np.random.seed(115)

orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean()

np.random.seed(10)
'''
 Reseed the singleton RandomState instance.
    
        Notes
        -----
        This is a convenience, legacy function that exists to support
        older code that uses the singleton RandomState. Best practice
        is to use a dedicated ``Generator`` instance rather than
        the random variate generation methods exposed directly in
        the random module.
'''
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10

############################
# Descriptive Statistics (Betimsel İstatistikler - Keşifçi Veri Analizi)
############################

df = sns.load_dataset("tips")
df.describe().T

############################
# Confidence Intervals (Güven Aralıkları)
############################
# Tips Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("tips")
df.describe().T

df.head()
#import statsmodels.stats.api as sms
sms.DescrStatsW(df["total_bill"]).tconfint_mean()
# (18.66333170435847, 20.908553541543164) => müşterilerin ödeyeceği hesap ortalaması istatistiki olarak %95 güven ile
# 18.66 ve 20.90 aralığındadır, %5 de hata payım vardır diyebiliriz.

sms.DescrStatsW(df["tip"]).tconfint_mean()
#(2.8237993062818205, 3.172758070767359)

# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T

sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()

######################################################
# Correlation (Korelasyon)
######################################################
# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

df = sns.load_dataset('tips')
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill")
plt.show()

# corr() => korelasyon alan method
df["tip"].corr(df["total_bill"])

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
# A bir grubu , B diğer grubu temsil eder. Bunlar arası farklılıkları inceleyen bir yöntemdir.
######################################################
# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara.Normallik sağlanıp Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İstatistiki OlAn Fark var mı?
############################
df = sns.load_dataset("tips")
df.head()

df.groupby("smoker").agg({"total_bill": "mean"})

############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2 (hesap ortalamaları arasında fark yoktur savı)
# H1: M1 != M2

############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı
# Varyans Homojenliği

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

# shapiro() => bir testin dağılımının normal olup olmadığını test eder.
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9367, p-value = 0.0002 => H0 RED oldu, non-parametrik kullanacaz

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9045, p-value = 0.0000 => H0 RED oldu,non-parametrik kullanacaz

############################
# Varyans Homojenligi Varsayımı : levene Testi kullanılır.
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

# levene() => iki testin dağılımının  olup olmadığını test eder.
test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 4.0537, p-value = 0.0452 => H0 RED oldu, varyanslar homojen değilmiş.

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)
# Normallik ve Varyans homojenliği sağlanıyorsa => equal_var=true  girilir.
# Normallik sağlanıp Varyans homojenliği sağlanmıyorsa => equal_var=false girilir. Arka planda welch testini yapar.

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 1.3384, p-value = 0.1820 => H0 reddedilemez oldu.

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################

# mannwhitneyu testi => median kıyaslama testidir.
test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 7531.5000, p-value = 0.3413 => H0 Reddedilemez oldu.

# ONEMLI NOT: H0'ı reddetmekle yapacağımız hatayı biliriz [alfa değeri(p-value)] ama
# H1'i kabul etmekle yapılacak hatayı bilmeyiz.
# Dolayısıyla H1 için bir dayanağımız olmadığından dolayı H0 temelli ilerleriz hep.

############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################
df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})
#sex
#female 27.91571
#male   30.72664

# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)

# 2. Varsayımları İncele
# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır


test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Varsayımlar sağlanmadığı için nonparametrik

# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Normallik sağlanmamıştı burada ekstradan baktık ama Varsayımlar sağlandığını gördük

# Non-Parametrik Testi Uygulayalım
test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 53212.5000, p-value = 0.0261 => H0 red

############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df = pd.read_csv("datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})
# Outcome = 0 => kişi diyabet değil
# Outcome = 1 => kişi diyabet

# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Normallik varsayımı sağlanmadığı için nonparametrik.

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("datasets/course_reviews.csv")
df.head()

df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()

test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################
# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)  # (3.7857863233209255, 0.0001532232957772221)
# p-value => 0.0001532232957772221

basari_sayisi / gozlem_sayilari

############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2 => p1 - p2 = 0 şeklinde de görebiliriz.
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()

df.loc[df["sex"] == "male", "survived"].mean()

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 16.2188, p-value = 0.0000

######################################################
# ANOVA (Analysis of Variance)
######################################################
# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.

df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()
#Thur   17.68274
#Fri    17.15158
#Sat    20.44138
#Sun    21.41000

# 1. Hipotezleri kur
# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.
# H1: .. fark vardır

# 2. Varsayım kontrolü
# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)


# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# 3. Hipotez testi ve p-value yorumu

# Hiç biri normal dağılımı sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean", "median"]})


# Hatırlayalım! =>  HO: Grup ortalamaları arasında ist ol anl fark yoktur

# parametrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])
# KruskalResult(statistic=10.403076391437086, pvalue=0.01543300820104127) => H0 red. İstatistiki anlamlı fark var gruplar arası.

## Farklılık Hangi Gruptan kaynaklanıyora cevap:
from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.09)
print(tukey.summary())