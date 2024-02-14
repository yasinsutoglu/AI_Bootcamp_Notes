print("hello world")
print("hello AI ERA")

####################################################
# Sayılar (Numbers) ve Karakter Dizileri (Strings)
####################################################
print(9)  #int
9.2       #float
type(9.2)
type("mrb")

####################################################
# Assignments & Variables
####################################################
a = 9
b = "mrb televole"
c = 10.2
d= a*c

####################################################
# Virtual Environment & Package Management
####################################################

### Tüm Bu Aşağıdakileri Terminal Local pencerede yazarız ###

#sanal ortamların listelenmesi => conda env list
#sanal ortam olusturma => conda create -n env_Name
#dışarıdan gelen yaml dosyası(aynı dizine path'ine dikkat) ile sanal ortam olusturma => conda env create -f environment.yaml
#sanal ortam silme => conda env remove -n env_Name
#sanal ortam aktive etme => conda activate env_Name ;  tersi de => conda deactivate
#ortam içi yüklü paketlerin listelenmesi => conda list
#paket yükleme => conda install numpy
#birden fazla paketi aynı anda yükleme => conda install numpy scipy pandas
#paket silme => conda remove package_name
#Belirli bir versiyona göre paket yükleme => conda install numpy=1.20.1 veya pip install numpy==1.20.1
#paket yükseltme => conda upgrade numpy veya conda upgrade -all
# version listesi ve dependency dısarı aktarma => conda env export > environment.yaml

#virtual env. => İzole çalışma ortamları oluşturmak için kullanılan araçlardır.
# Farklı çalışmalar için oluşturulabilecek farklı kütüphane ve versiyon ihtiyaçlarını çalışmalar birbirini
# etkilemeyecek şekilde oluşturmaya imkan sağlar.

#Dependency Managment Tool => config file'da dependency ayarları ekleme aracı









