############################################
# User-Based Collaborative Filtering
# User benzerlikleri üzerinden öneriler yapılır. X kişisinin kullanıcı davranışlarıyla Y kişisinin
# kullanıcı davranışları birbirine benzer olsun. Dolayısıyla biz X kullanıcısına önereceğimiz
# filmleri Y kullanıcısının beğenilerini baz alarak belirleyebiliriz. Diğer yöntemlerde
# sadece bir kişinin bir şeye temas etmesi ile ilgileniliyordu. Burada bir kişiye daha detaylıca odaklandık.
# Yani burada ihtiyaç duyulan veri daha özel ve kişisel veridir. Konsept budur.
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('section_datasets/recommender/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('section_datasets/recommender/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 10000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

import pandas as pd
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# random_state=45 => hoca ile aynı kullanıcıyı seçmek için setlendi.

#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################
#random_user
#user_movie_df

random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# burada sağlama yapıp görmek için yazdık, örnek çeşitlendirmek için film ismini değiştir.
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]


len(movies_watched)

#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

movies_watched_df = user_movie_df[movies_watched]

user_movie_count = movies_watched_df.T.notnull().sum()

#transpoze sonrası userId'leri index olmaktan kurtarıp asıl index değerlerini verme işlemi:
user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

# tavsiye vereceğimiz kullanıcı ile en az 20 film ortak izlemiş kullanıcılar
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# tavsiye vereceğimiz kullanıcının izlediği tüm filmleri izlemiş olan kullanıcıların sayısı
user_movie_count[user_movie_count["movie_count"] == 33].count()

# 20 => burada kendi yorumumuza göre belirlemiş olduğumuz eşik değeridir.
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################
# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan(tavsiye vereceğimiz kullanıcı) ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

# 1.nci basamak
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

# final_df korrelasyon sonrası pivotunu aldık
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

# Sinan ile en az %65(kendi yorumumuzla eşik değeri) korrelasyona sahip kullanıcıları getirdik
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)


rating = pd.read_csv('section_datasets/recommender/datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################
# burayı son tablomuzdan yorumlayarak corr ve rating değerlerinin trade-off'unda kararsız
# kalınca her zaman yaptıgımız weighted score ile işi çözelim dedik.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

# yine eşik değeri belirleme ile eleme yaptık
recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('section_datasets/recommender/datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])

#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)


