#####################################
# FUNCTIONS - Belirli işlevleri yerine getirmek için yazılan kod bloklarıdır.
####################################
# ?print =>  "builtin_function_or_method" olarak sonucu gördük
# parameter =>  fonksiyon tanımlanması esnasında ifade edilen değişkenler
# argument =>   fonksiyon çağırıldığında parameter değerlerine karşılık girilen  değerlerdir. Fonksiyon genel amacını biçimlendirmek
# üzere kullanılan alt görevcilerdir. Docstring => fonksiyon dökümantasyonu
print("a", "b", sep="-")  # help(print) => console'a yazınca docstring elde ederiz


# ------------Function Definition---------------------
# Fonksiyon body kısmı => "statement" kısmıdır.
# def function_name(parameters/arguments):
#	statements (function body)

def fonksiyonAdi(params1, params2):
	print(params1 * params2)


fonksiyonAdi(3, 5)


# Example -1
def summer(arg1, arg2):
	print(arg1, arg2)
	return arg1 + arg2

# RETURN => fonksiyon çıktılarını girdi olarak kullanmak amacına hizmet eder. Fonksiyon sonucunda ne elde edeceğizin cevabıdır. Fonksiyon
# return'u gördükten sonra çalışmayı keser.

result = summer(5, 5)
print(result)

summer(arg2=8, arg1=3)
#NOT : argüman sırasını biliyorsan direkt sıralı girilebilir ama bilinmiyorsa "summer(arg2=8, arg1=3)" buradaki gibi argüman isimlerine
# direkt assignment yapılabilir.

#-----------------DocString----------------------------
# Settings =>  DocString Search Et => Tools altından gir => Docstring Format=Numpy veya Google
# """ koy Enterla
# help(multiply) => func. detay bilgilere erişir.

def multiply(arg1, arg2):
	"""
	DOCSTRING =>Fonksiyonlarımıza herkesin anlayabileceği ortak bir dil ile bilgi notu ekleme yoludur.
	İlk kısım; fonksiyon işlevi anlatılır.
	İkinci kısım; parametreler hakkında bilgidir.
	Üçüncü kısım; Çıktısının ne olacağı bölümüdür.
	-------------------------------------
	:param arg1: int, float
		( arg1 görevi eklenebilir)
	:param arg2: int, float
	-------------------------------------------
	:return: int, float
	"""
	return arg1 * arg2


x = multiply(5, 3) / 10
print(x)

#----------------------- Fonksiyon Statement/Body Bölümü ----------------------
# Fonksiyonun ne yapacağına ve hangi sıra ile yapacağına karar verip Python'a ifade ettiğimiz kısım.
# Body kısmı =>  local scope; dışarısı global scope'tur.

def say_hi():
	print("hi")
	print("hello")
	print("merhaba")

say_hi()

#--------------------------------------
# list_store =>  global değişken , c => local değişken
#global değişken, local scope içerisinde de değiştirilebilir.

list_store = []
def ekle_listeye(par1, par2):
	c = par1 * par2
	list_store.append(c)
	print(list_store)


ekle_listeye(1, 8)
ekle_listeye(3, 6)
ekle_listeye(10, 7)

#---------------Ön Tanımlı(Default) Arguments/Parameters: Fonksiyon tanımlarken değer atamsı yapmaktır---------------
def divide(a = 5 , b = 2):
	print(a/b)


divide(1, 2)    #sonuc =>0.5
divide()              #sonuc =>2.5
divide(3)             #sonuc =>1.5

#------------------RETURN kullanımı--------------------------
def calculate(warm, moisture, charge):
	warm = warm * 2
	moisture = moisture * 3
	charge += 2
	output = (warm + moisture) / charge

	return warm, moisture, charge, output

calculate(98,12,78)

type(calculate(98,12,78)) # tuple olarak gösterir

a, b, c, d = calculate(98,12,78)
print(a, b, c, d) # çıktı => 196 36 80 2.9

#------------------Fonksiyon içinde Fonksiyon Çağırmak--------------------------

def my_div(a = 5 , b = 2):
	out = a / b
	return int(out)

def standardization(k = 2,  m=3):
	return k * 10 / (m + m)

def all_calc(a, b, p):
	c = my_div(a , b)
	d = standardization(c, p)
	print(2 *  d)

#I. Global Scope: Global etki alanında olan bir değişkendir ve programın herhangi bir bölümünden erişilebilir.
#II.Local Scope: Bir değişkenin veya fonksiyonun local etki alan tanımlandığı yerde veya daha özel bir kapsam içinde erişilebilir olduğu
# alandır.