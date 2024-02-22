#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
# Ürün içeriklerinin(metadata) benzerlikleri üzerinden tavsiyeler geliştirilir.
#############################
# örn metadatalar => film açıklaması/yönetmen/oyuncu kadrosu, bir kitabın açıklaması/özeti, bir ürün açıklaması/kategori bilgisi

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("section_datasets/recommender/datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)
# DtypeWarning kapamak icin
df.head()
df.shape

df["overview"].head()

tfidf = TfidfVectorizer(stop_words="english") #the , on gibi ara kelimeleri analizimizi saptırmasın diye çıkararak tdidf yaptık

# df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('') # acıklaması boş olanları çıkardık

tfidf_matrix = tfidf.fit_transform(df['overview'])

tfidf_matrix.shape #(45466, 75827) => ilki yorumlar, ikincisi unique kelimeler

df['title'].shape

tfidf.get_feature_names_out()
#tfidf.get_feature_names()

csr_tfidf_matrix = csr_matrix(tfidf_matrix, dtype=np.float32)
# tfidf_matrix.toarray() => Unable to allocate 25.7 GiB for an array with shape (45466, 75827)
# and data type float64 için memoryError verdi ve ben de "from scipy.sparse import csr_matrix" kullandım
csr_tfidf_matrix.toarray()

#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

# diğer benzerlik/uzaklık ölçüleri => manhattan distance, öklid distance, pearson correlation etc.
# durumdan duruma göre hangisini kullanacağımın tercihi değişir.
cosine_sim = cosine_similarity(csr_tfidf_matrix,
                               csr_tfidf_matrix)

cosine_sim.shape
# (45466, 45466) => 45466: overview'ların sayısıdır
cosine_sim[1] # bu 1nci indexteki filmin diğer tüm filmlerle benzerlik score'ları var

#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

# pandas series olusturup dataframe index sütunu ve title sütunu index bilgilerini verdik
indices = pd.Series(df.index, index=df['title'])

indices.index.value_counts()
#title
#Cinderella              11
#Hamlet                   9
#Alice in Wonderland      9
#Beauty and the Beast     8
#Les Misérables           8
#                       ..
# Burada bazı titlelar birden fazla matriste yer aldığı için teke düşürmemiz gerekir.
# Bu bir sorundur.Bu tip çoklanan isimlendirmelerin en sonuncusu alınır.
# duplicated(keep='first') default'tur. True/ false döner. False => duplicate olmayandır.

indices = indices[~indices.index.duplicated(keep='last')]

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index] # "Sherlock Holmes" filmi ile diğer filmlerin similarity'leri ölçüldü

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
# 0ncı indexte kendi filmimiz var oldugu için atladık ve [1:11] yaptık

df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
# Bu işlemleri webSitemizdeki en çok izlenen 100 filmi çekip her film için öneri listesi olusturup SQL tablosuna gömeriz
# Kullanıcı bu 100 filmden birini izleyince tak otomatik tablodan çekip onune basarız.
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
