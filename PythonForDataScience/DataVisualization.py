########################
# VERI GORSELLESTIRME : MATPLOTLIB & SEABORN (high-level)
#########################
# Matplotlib => python veri görselleştirme araçlarının atası (low-level)

# Kategorik değişken => sütun grafik veya piechart ile görselleştir. Countplot ve Barplot
# Sayısal değişken => Histogram ve Boxplot kullan.
# PowerBI, Tableu veya ClickView daha uygun araçlardır aslında görselleştirme konusunda. (VeriTabanı ile direkt bağlantılıdırlar)

# Categoric Variables(text or string etc) Visualization #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt #pip install --upgrade matplotlib yapabiliriz.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show()

# Numerical Variables Visualization #
plt.hist(df["age"]) # elimizdeki sayısal değişkenin belirli aralıklara göre dağılım bilgisini verir
plt.show()

plt.boxplot(df["fare"]) # dağılım bilgisi ve aykırı değerleri görmeye yarar
plt.show()

###################
# MATPLOTLIB FEATURES
###################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# 1) PLOT-----------
x = np.array([1, 8])
y = np.array([0, 150])
plt.plot(x,y)
plt.plot(x,y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])
plt.plot(x,y)
plt.plot(x,y, 'o')
plt.show()

# 2) MARKER-----------
y = np.array([13, 28, 11, 100])
plt.plot(y, marker= 'o')
#plt.plot(y, marker= '*')
#markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']
plt.show()

# 3) LINE-----------
y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle= "dashed", color="r") # dotted, dashdot
plt.show()

# Multiple Lines
x = np.array([23, 47, 61, 89, 110])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

# 4) Labels-----------
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
plt.title("Bu ana başlık")
plt.xlabel("x ekseni")
plt.ylabel("y ekseni")
plt.grid()
plt.show()

# 5) Subplots-----------

#plot1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(2,2,1)
plt.title("1")
plt.plot(x , y)

#plot2
x2= np.array([23, 47, 61, 89, 110])
y2= np.array([13, 28, 11, 100, 117])
plt.subplot(2,2,2)
plt.title("2")
plt.plot(x2 , y2)

#plot3
x3 = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y3 = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(2,2,3)
plt.title("3")
plt.plot(x3 , y3)

###################
# SEABORN
###################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df = sns.load_dataset("tips")
df.head()

# Kategorik Değişken Görselleştirme
df["sex"].value_counts()
sns.countplot(x= df["sex"], data=df)
plt.show()

#Sayısal Değişken Görselleştirme
sns.boxplot(x = df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()