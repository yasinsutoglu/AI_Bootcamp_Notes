###########################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
###########################
# - NumPy (numerical python) -1995 python kurucularının olusturdugu math & stats lib.
# - Pandas (Panel Data Analysis) => veri manipulasyonu deyince akla gelir. Numpy özelliklerini geliştirmiştir.
# - Veri Görselleştirme: Matplotlib (low-level) & Seaborn(High-Level)
# - Gelişmiş Fonksiyonel KEşifçi Veri Analizi (Advanced Funcitonal Exploratory Data Analysis)

#############################
# NUMPY => array'ler, çok boyutlu array'ler ve matrisler üzerinde yüksek performansla çalışma imkanı sağlar. Listelerden farkı verimli
# veri saklama ve (yüksek seviyeden işlemler)vektörel operasyonlardır. Döngü yazmaya gerek kalmadan işlem yapabilme olanağı sağlar.
# Neden Numpy ? => daha az çaba ile daha fazla işlem yapma
################################

import numpy as np
a=[1, 2, 3, 4]
b=[2, 3, 4, 5]

#python ile yapılış
ab= []
for i in range(0,len(a)):
	ab.append(a[i] * b[i])

#numpy ile yapılış
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

##############################
# Creating Numpy Arrays => numpy array'i python veri yapıları gibi bir veri yapısıdır. Genelde 0'dan olusturulmaz, çalışmamızda bulunan
# verilerden dönüştürülerek elde edilir.
###############################
import numpy as np

a = np.array([1, 2, 3, 4, 5])
type(a) #  numpy.ndarray

np.zeros(4) # array([0., 0., 0., 0.])
np.zeros(10, dtype=int)

np.random.randint(0, 12, size=5) # array([9, 7, 8, 5, 9])
np.random.normal(10 , 4 ,(3, 4)) # mean:10 , std: 4, 3x4'luk normal dağılımlı sayılardan olusan array yaptık
# random module araştırılabilir daha fazla bilgi için

###########################
# Attributes of Numpy Arrays
# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi
##########################
import numpy as np
a = np.random.randint(2, 10, size=5)
print(a) # [2 3 6 5 8]
print(a.ndim) # 1 dimension
print(a.shape) # (5,) => tek boyutlu ve içinde 5 eleman var
print(a.size) # 5
print(a.dtype) #int32

######################
# ReShaping => elimizdeki numpy array'inin boyutunu değiştirmek istersek kullanırız.
######################

np.random.randint(1 , 10, size=9)
np.random.randint(1 , 10, size=9).reshape(3,3)
# 3x3'luk matrise cevirdik
# array([[1, 4, 4],
#      [4, 1, 9],
#      [3, 5, 5]])

ar = np.random.randint(1 , 10, size=12)
ar.reshape(3,4) #resahpe ederken matris eleman sayısı ve array eleman sayısı eşit olmalı dikkat!

############################
# Index Selection (Indeks Seçimi)
############################

a =  np.random.randint(10, size=10) # array([1, 7, 7, 0, 9, 2, 5, 3, 1, 5])
a[0] = 999 # yapabiliriz
a[0:5] # array([1, 7, 7, 0, 9]) => slicing

m = np.random.randint(10, size=(3 , 5))
m[0][3] # ilki satır, ikinci sütun
m[0, 3] # <= m[0][3]
m[1, 1] = 14
m[2, 3] = 2.9 # 2 değerini atar çünkü numpy, "fixed type" veri tutar ondan integer'a cevirip aldı
m[:, 0] # tüm satırların ilk sütun değerlerini al => array([2, 0, 9])
m[1, :] # ilk satırın tüm sütunlarını al
m[0:2, 0:3] # 0 ve 1nci satır ; 0-1-2nci sütun değerleri alınır.

############################
# Fancy Index => bir takım kod akışı esnasında elimde birden fazla index var; bu minvalde, bir numpy arrayıne bir liste(index number yada
# true/false olabilir) girilince bize kolay bi sekilde seçim işlemi sağlar.
############################

v = np.arange(0, 30, 3) # 0'dan 30a kadar (30 haric) 3erli artacak olusan array
v[4]

catch = [1, 2, 3]

v[catch] # catch'teki indexlere karşılık gelen değerleri v array'inden alır gelir

############################
# Conditions on Numpy => arka planda fancy index kullanılır.
############################

y = np.array([1, 2, 3, 4, 5])
y < 3 # array([ True,  True, False, False, False])
y[y < 3 ] # array([1, 2]) => true'ya karşılık gelenler seçildi
y[y != 3 ] # array([1, 2, 4, 5])

############################
# Mathematical Operations
############################

k = np.array([1, 2, 3, 4, 5])
k / 5 # array([0.2, 0.4, 0.6, 0.8, 1. ]) => tüm elemanlara bu operandı uygular
k ** 2 #array([ 1,  4,  9, 16, 25])

np.subtract(k, 1) # <= (k - 1)
np.add(k, 1)
b = np.mean(k) # ortalama
np.sum(k)  # 15
np.min(k)
np.max(k)
a = np.var(k) # varyans

# trigonometrik fonks, türev, integral de yapılabilir.İki bilinmeyenli math equations da yapılır.

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

z = np.array([[5, 1], [1, 3]])
u = np.array([12, 10])

np.linalg.solve(z, u) # linalg => linear algebra
#array([1.85714286, 2.71428571]) => (x0 , x1)