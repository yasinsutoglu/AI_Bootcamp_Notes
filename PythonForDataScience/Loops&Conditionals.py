#################################
# CONDITIONS => program yazımı esnasında akış kontrolü sağlayan yapılardır. Programın belli koşullara göre nasıl hareket etmesini belirten
# kodlardır.
# if , elif , else yapıları kullanılır. İç kısımları da local scope ve statement kısmıdır.
#################################

if 1 == 1:
	print("do something")

#-------------------------------
def number_check(number):
	if number > 10:
		print("sayı 10dan buyuk")
	elif number < 10:
		print("sayı 10dan küçüktür")
	else:
		print("sayı 10dur")

number_check(10)
number_check(12)

#############################################
# LOOPS - üzerinde iterasyon yapılabilen nesneler üzerinde gezinmeyi ve bu gezinme sonucunda yakaladığımız her eleman üzerinde çeşitli
# işlem yapabilme yetisi veren kod yapılarıdır.
# for loop
##########################################

students = ["john", "mark", "venessa", "mariam"]
salaries= [1000, 2000, 3000, 5000, 10000]

for element in students:
	print(element)

for i in students:
	print(i.upper())

for sal in salaries:
	print(int(sal*20/100 + sal))

#------Example---------
def new_salary(sal , a):
	return int(sal * a/100 + sal)

for sal in salaries:
	if sal > 3000:
		print(new_salary(sal, 30))
	else:
		print(new_salary(sal, 50))

#------------------GUZEL SORU--------------------------------

#range() =>  iki değer arası sayı üretmeye yarar
# range(len("miuul")) veya range(2,7) gibi kullanımı var(7 hariç)

def alter(string):
	new_str = ""
	for index in range(len(string)):
		if index % 2 == 0:
			new_str += string[index].upper()
		else:
			new_str += string[index].lower()
	return new_str

sonuc = alter("soKAkTA sayaMAm gibİ")
print(sonuc)

#######################################################################
#-----------------------BREAK & WHILE & CONTINUE-----------------------

my_salaries= [1000, 2000, 3000, 5000, 10000]

for sal in my_salaries:
	if sal == 3000:
		break
	print(sal)

# ---------------------

for sal in my_salaries:
	if sal == 2000:
		continue
	print(sal)

#--------------------

num = 1
while num < 5:
	print(num)
	num += 1

#######################################################################
#ENUMERATE : Otomatik Counter/Indexer ile "for loop" ; Liste ve String'ler için çalışır.
# ---------------------------------------------------------

students = ["john", "mark" , "venesssa", "meriam"]

for index, std in enumerate(students):
	print("index:", index)
	print("student:", std)

#--------------- Nice Example ---------------------
def div_std(stdnts):
	groups = [[], []]
	for i , std in enumerate(stdnts):
		if i % 2 == 0:
			groups[0].append(std)
		else:
			groups[1].append(std)
	return groups

result = div_std(students)
print(result) # [['john', 'venesssa'], ['mark', 'meriam']]

#######################################################################
#ZIP : farklı liste türlerini index bazlı olarak tuple formatında birleştirir.

students = ["john", "mark" , "venesssa", "meriam"]
my_salaries= [1000, 2000, 3000, 5000, 10000]

print(list(zip(students, my_salaries))) # [('john', 1000), ('mark', 2000), ('venesssa', 3000), ('meriam', 5000)]

#######################################################################
#LAMBDA & MAP & FILTER & REDUCE => vektor seviyesine işlem yapmaya yararlar
#lambda en önemlisi ve apply() fonksiyonu ile kullanılır genelde.
#lambda, kullan at fonksiyon tipidir.Yani değişkenler bölümünde yer/kayıt tutmaz.
# Fonksiyon ismi tanımlamadan kullanılır. JS'teki IIFE(immediately Invoked Function Expression) gibi.
#######################################################################

new_sum = lambda a, b: a + b

print(new_sum(3,5))

#------------------------------
#map(kullanılacakFonkAdi, listeAdi) => js map() gibi ; list() => liste olusturma seklidir.

students = ["john", "mark" , "venesssa", "meriam"]

def largen(x):
	return x.upper()

new_list = list(map(largen , students ))
print(new_list) #sonuc: ['JOHN', 'MARK', 'VENESSSA', 'MERIAM']

#-------------

list2 = list(map(lambda x: x.upper() , students)) # lambda fonk =>  javascript'teki arrow function gibi kullanıldı
print(list2) # sonuc: ['JOHN', 'MARK', 'VENESSSA', 'MERIAM']

# FILTER ---------------------------

list_store = [1,2,3,4,5,6,7,8,9,10]
list(filter(lambda x : x % 2 ==0, list_store))

# REDUCE ---------------------------
from functools import reduce
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
reduce(lambda x,y : x + y , list_store)
