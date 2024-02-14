############################
# DATA STRUCTURES
#########################
# - Sayılar (Numbers) => int, float , complex
# - Karakter dizileri(Strings) => str
# - Boolean (True-False) => bool
# - Liste (List)
# - Sözlük (Dictionary)
# - Demet (Tuple)
# - Set

#! SESSION START
x = 46
type(x)

y=10.3
type(y)

z = 2j +1
type(z)

m = "Hello ai era"
type(m)

type(True)
5 == 4
1 == 1
type(3==5)

#List
k = ["a","b","c"]
type(k)

#Dict (key-value çifleri kullanılır)
n = {
    "name":"peter",
    "Age":37
}
type(n)

# Tuple
t = ("python", "ml" , "ds")
type(t)

#Set
s = {"yasin", "busra", "citir"}
type(s)

# NOT : List , tuple , set ve dict veri yapıları aynı zamanda Python Collections (Arrays) olarak geçmektedir.

#-----MATHS--------
yas = x * 3
mim = x / 7
x * y / 10
x ** 2 # kare alma
print(yas)

# --------TYPE CASTING/COERSION------------
print(int(10.5))
print(float(5))
print(int(mim))

# STRING
print("John")
print('John')

name = 'Boss ramiz'
print(name)

var1 = """Çok satırlı karakter dizisi olarak kullanılabilir"""
var2 = '''Adrenocorticotropic hormone is a polypeptide tropic hormone produced by and secreted by the anterior pituitary
gland. It is also used as a medication and diagnostic agent. ACTH is an important component of the hypothalamic-pituitary-adrenal
axis and is often produced in response to biological stress. Its principal effects are increased production and release of 
cortisol and androgens by the cortex and medulla of the adrenal gland, respectively. ACTH is also related to the circadian 
rhythm in many organisms.'''

ilkHarf = var1[5]
print(ilkHarf)

# String Slicing
print(var1[0:15])

#Stringte Eleman sorgulama
print("Çok" in var1) #true
print("cok" in var1) #false

#----------STRING METHODS (Method => Class içinde tanımlı fonksiyonlar)-----------
dir(str)  # string ile ilgili methodları görmek için kullanırız
dir(5)
dir("yaso")

name1 = "john"
type(len) #builtin_function
print(len(name1))

print("miuul".upper())
print("YASO".lower())
# type(upper), type(upper()) => herhangi birsey vermez çünkü bu builtin_method. Üzerine gelip "CTRL + left click" ile detayına gideriz

hi = "Hello AI Era"
print(hi.replace("e", "o"))

print("Hello AI Era".split("A")) # "A"ya göre parçalara ayırır => ['Hello ', 'I Era']
print("Hello AI Era".split()) #default " "a göre böler =>['Hello', 'AI', 'Era']

print(" ofofof ".strip()) #boslukları kırpar => ofofof
print("ofofo".strip("o")) #"o"ları kırpar => fof

print("foo".capitalize()) # baş harf büyük

print("foo".startswith("f")) #true doner

# -----------------LIST  ----------------------
# Değiştirilebilir
# Sıralıdır. Index işlemleri yapılabilir
# Kapsayıcıdır.İçerisinde birden fazla veri yapısını tutabilir.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b" , "c" , "v" ]
mix_types = [1, 2, 3, "a", "b", True, [1, 2, 3]]
print(mix_types[6])
print(mix_types[6][1])

mix_types[0] = 99
print(mix_types[0])

print(mix_types[0:4]) # SLICING

# -------------LIST METHODS ------------
dir(list) #  'append', 'clear',  'copy',  'count',  'extend',  'index',  'insert',  'pop',  'remove',  'reverse',  'sort'

# len :  builtin python fonksiyonu , boyut bilgisi
len(mix_types)

mix_types.append({"a" : 1}) #sona eleman ekler
mix_types.pop(3) # index'e göre eleman siler
mix_types.insert(4, (3, 4, 5)) # index'e eleman ekler

#----------DICTIONARY-----------
# Değiştirilebilir
# Sırasız (python 3.7 sonrası sıralı)
# Kapsayıcı

# KEY-VALUE çifti şeklinde eleman eklenir.

my_dictionary = {
    "REG": "regression",
    "2": "Yasin",
    "LOG": "Cmy_cart",
    "liste": ["asd", 12],
    "num": 13,
    "dict": {
        "a": (1,23, 44)
    },
}

my_dictionary["REG"] # 1.yöntem: value çağırma işlemi yaptık
my_dictionary.get("REG") # 2.yöntem: value çağırma işlemi yaptık

my_dictionary["dict"]["a"]
my_dictionary["liste"][1]

# Key Questioning (Sorgulama) => dictionary içinde key'e göre var mı yok mu sorgusu
"REG" in  my_dictionary # true doner

my_dictionary["liste"] = [55 , 14]

# TÜM KEY'lere ERİŞMEK
dir(my_dictionary) # 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values'

my_dictionary.keys()
my_dictionary.values()
my_dictionary.items() # tüm çiftler tuple formatında elde edilir => [('REG', 'regression'), ('2', 'Yasin'), ('LOG', 'Cmy_cart'), ('liste', [55, 14]), ('num', 13), ('dict', {'a': (1, 23, 44)})]

# KEY- VALUE  değeri Güncellemek
my_dictionary.update({"REG": 1923})

#Yeni Key-Value (en sona) Eklemek
my_dictionary.update({"AAA": "olmayan key ile update kullanım"})

# ----------------TUPLE (demet)-------------------
#listenin kardeşidir
# DEĞİŞTİRİLEMEZ => DOES NOT SUPPORT item assignment
# Sıralıdır
# Kapsayıcıdır
# kullanım sıklıgı azdır. Senaryoya göre kullanılabilir.

my_tuple = ("john", "mark", 1, 2)
my_tuple[0]
my_tuple[0:3] # slicing

# değiştirmek için arka yoldan dolanırız. Type Casting ile yaparız.
my_tuple = list(my_tuple)
my_tuple[0] = 11
my_tuple = tuple(my_tuple) # (11, 'mark', 1, 2) son elde edilendir.

# ------------- SET (kümeler gibi düşünülebilir)-----------------------
# Değiştirilebilir
# SIRASIZ + EŞSİZ
# Kapsayıcıdır
# Hız gerektiren , küme işlemleri gibi (kesişim, birleşim, fark) işlemler gerekince kullanırız!
# Data Science içinde kullanımı düşüktür.

#-------------------
set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

# difference() : iki küme farkı
set1.difference(set2)  #set1'de olup set2'de olmayanlar => {5}
set1 - set2

# symmetric_difference()  : iki kümede birbirine göre olmayanlar
set1.symmetric_difference(set2) # set1, set2'de birbirine göre olmayanları getirme => {2,5}

# intersection() : iki küme kesişimi
set1.intersection(set2) # {1,3}
set1 & set2 # {1,3}

# union() : iki küme birleşimi
set1.union(set2) # {1, 2, 3, 5}

# isdisjoint() : iki küme kesişimi boş mu??
set1.isdisjoint(set2) # False

# issubset() : bir küme diğerinin alt kümesi mi??
set1.issubset(set2) # False

# issuperset() : bir küme diğerini kapsar m??
set1.issuperset(set2) # False



























