#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

# eldeki film listesinden örnek veri seti oluşturduk
sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape #(76918,4) => 76918 : kullanıcıları ifade ediyor bize; 4 : filmleri ifade eder

reader = Reader(rating_scale=(1, 5)) # belirleyeceğimiz skala aralığını olusturduk

# skala üzerinden ratingli hale getirdik datayı
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Adım 2: Modelleme
##############################

#ML olaylarında veri setlerini trainset ve testset diye ayırıp, modeli bir eğitim seti üzerinde kurup sonra
# modeli daha önce görmemiş olan test seti üzerinden test ederiz konseptini hatırlayalım.
trainset, testset = train_test_split(data, test_size=.25) # %75 trainset, %25 testset olarak böl.
svd_model = SVD() # matrix fact. için kullanılacak fonksiyondur.
svd_model.fit(trainset) # model kurma işlemi yapıldı. Trainset üzerinden öğren dedik.(p ve q ağırlıkları bulduk)
predictions = svd_model.test(testset) #blank tahminleri yapılır.

accuracy.rmse(predictions) # hata kareler ortalaması karekökü alınır (minimizasyon için)
# burada yapmam beklenen ortalama hata ortaya çıkar.

# spesifik kullanıcı için tahmin yaptık
svd_model.predict(uid=1.0, iid=541, verbose=True)
svd_model.predict(uid=1.0, iid=356, verbose=True) # estimated rating=4.16
sample_df[sample_df["userId"] == 1] # real rating value = 4.0

##############################
# Adım 3: Model Tuning => Temel modeli optimize etmek işlemidir. Model tahmin performansı arttırma biryerde yani.
# Modelin dışsal/kullanıcı müdahelesine açık/hiperparameter olan parametreleri nasıl optimize edeceğimiz mevzusu söz konusudur.
##############################

# epoch sayısı => iterasyon sayısı (number of iteration of the Stochastic Gradient Descent procedure)
# lr_all => learning rate (gamma semboldü hatırla!)
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}
# hiperparametreler; SVD() fonksiyon detayında yer alan Args'lerdir.

# 'rmse', 'mae'(mutlak hata ortalaması) => error değerlendirme ölçütleri
gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3, # 3 katlı çapraz doğrulama(cross validation) yapmak : verisetini 3e böl 2 parça ile model kur
                  # 1 parça ile test et. Bunu kombinatorik tekrarla! Bu test işlemlerinin ortalamasını al!
                  n_jobs=-1, #işlemcileri full performance kullan.
                  joblib_verbose=True) # işlem yapılırken raporlama yap!

gs.fit(data)

gs.best_score['rmse'] # 0.93.. çıktı
gs.best_params['rmse']

##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params['rmse']) # func_name(**kwargs) mevzusu var burada

data = data.build_full_trainset() # bütün veri seti full trainset oldu.
svd_model.fit(data)

# spesifik film("Blade Runner") için tahmin olusturduk
svd_model.predict(uid=1.0, iid=541, verbose=True) # est = 4.20.. ; realValue => 4.0 idi.






