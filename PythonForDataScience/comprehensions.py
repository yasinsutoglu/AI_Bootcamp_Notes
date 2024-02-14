#################################
#COMPREHENSIONS : Birden fazla kod satırına sahip kod yapılarını tek satırda yapmamıza olanak sağlayan yapılardır.
# 1. LIST COMPREHENSIONS
# 2. DICT COMPREHENSIONS
################################

############## LIST COMP. => çıktısı liste olan comprehension tipi #######################

# UZUN YONTEM
salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
	return x * 20 / 100 + x

null_list = []
for salary in salaries:
	if salary > 3000:
		null_list.append(new_salary(salary))
	else:
		null_list.append(new_salary(salary * 2))

# COMP. YONTEMI => IYI OGRENILMELIDIR
# if tek kullanıcaksa en sağda olur; if-else şeklinde kullanılacaksa "for" yapısı en sağda olur if-else ortada olur.
[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

[ salary * 2 for salary in salaries if salary < 3000] #cıktı : [2000, 4000]

# Example
students = ["John", "Mark", "Venessa", "Mariam"]
students_no = ["John", "Venessa"]

[std.lower() if std in students_no else std.upper() for std in students] #cıktı : ['john', 'MARK', 'venessa', 'MARIAM']
[std.lower() if std not in students_no else std.upper() for std in students]

############## DICT COMP. => çıktısı dictionary olan comprehension tipi #######################

my_dict = {
	'a': 1,
	'b': 2,
	'c': 3,
	'd': 4
}

my_dict.keys()
my_dict.values()
my_dict.items()

{k : v ** 2 for (k,v) in my_dict.items()} # cıktı : {'a': 1, 'b': 4, 'c': 9, 'd': 16}

{k.upper() : v  for (k,v) in my_dict.items()} # cıktı : {'A': 1, 'B': 2, 'C': 3, 'D': 4}

############# UYGULAMA ORNEKLERI ###########
# ORN_1
numbers = range(10)
new_dict = {}

for n in numbers:
	if n % 2 == 0:
		new_dict[n] = n ** 2 # burada new_dict[n] => n: key olarak eklenir, n ** 2 => value olarak eklenir

# kısayol
{n : n ** 2 for n in numbers if n % 2 == 0} #  {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# ORN_2 : Bir veri setindeki değişken isimlerini değiştirmek
# dataFrame =>  excel tablosu gibi
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
	print(col.upper())

df.columns = [col.upper() for col in df.columns]

# ORN_3 : İsminde 'INS' olan değişkenlerin başına FLAG, değerlerine ise NO_FLAG eklemek istiyoruz.

["FLAG"+col for col in df.columns if "INS" in col]

df.columns = ["FLAG" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns ]

# ORN_4 : Amacımız; key'i string, value'su aşağıdaki gibi bir liste olan dictionary oluşturmak. Bu işlemi sadece sayısal değişkenler için
# yapmak istiyoruz.

import seaborn as sns
df= sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"]
# O => Object (numerik değil, kategorik değişkeni belirtir)
# df[col] => dataframe ilgili kolon

soz = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
	soz[col] = agg_list

#kısayol
new_dict = {col: agg_list for col in num_cols}

#cıktı :
# {'total': ['mean', 'min', 'max', 'sum'],
# 'speeding': ['mean', 'min', 'max', 'sum'],
# 'alcohol': ['mean', 'min', 'max', 'sum'],
# 'not_distracted': ['mean', 'min', 'max', 'sum'],
# 'no_previous': ['mean', 'min', 'max', 'sum'],
# 'ins_premium': ['mean', 'min', 'max', 'sum'],
# 'ins_losses': ['mean', 'min', 'max', 'sum']}

df[num_cols].head()

df[num_cols].agg(new_dict)
# agg() fonksiyonu kolon isimleri eslesiyorsa agg içinde value olarak verilen fonksiyon listesinde tüm fonksiyoları sütuna uygular.