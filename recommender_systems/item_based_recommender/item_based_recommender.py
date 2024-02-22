###########################################
# Item-Based Collaborative Filtering
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması



######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('section_datasets/recommender/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('section_datasets/recommender/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId") # SQL'deki JOIN işlemi yaptık.
df.head()


######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################
# Örneğin 3 filmden biri 1000 rating , diğeri 40 rating, öbürü 3 rating almış olsun
# Biz bu az rating alanları hesaba katarsak hesaplama maliyetleri külfetli olur
# Dolayısıyla bu tarz filmleri burada eleme işlemi yapıyoruz.
df.head()
df.shape

df["title"].nunique()

df["title"].value_counts().head() # hangi filme kacar yorum gelmiş

comment_counts = pd.DataFrame(df["title"].value_counts())
#                                           count
#title
#Pulp Fiction (1994)                        67310
#Forrest Gump (1994)                        66172
#Shawshank Redemption, The (1994)           63366
#Silence of the Lambs, The (1991)           63299

rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape # (17766015, 6)

common_movies["title"].nunique() #3159 tane film 17766015 tane rating almış.
df["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
# DÜZELTME HATA ALMAMAK İÇİN
#common_movies = common_movies.dropna()
#common_movies_cleaned = common_movies.dropna().drop_duplicates(["userId", "title"])
#common_movies_cleaned.loc[:, "rating"] = common_movies_cleaned["rating"].astype("float32", copy=False)
#common_movies_cleaned.loc[:, "userId"] = common_movies_cleaned["userId"].astype(str,errors='ignore',copy=False)
#common_movies_cleaned.loc[:, "userId"] = common_movies_cleaned["userId"].str.split(".", expand=True)[0]
#user_movie_df = common_movies_cleaned.pivot_table(index="userId", columns="title", values="rating")

user_movie_df.shape
user_movie_df.columns


######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]
# corrwith() => korrelasyona bakma fonksiyonu
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

## RASTGELE BİR FILM SECIP ONUNLA CORRELATION'a bakma
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

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


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)





