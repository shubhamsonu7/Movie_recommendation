# ============================================================
#             MOVIE RECOMMENDATION ENGINE
#             AIML MINOR PROJECT
#
#             Technique:
#             Content-Based Filtering
#             TF-IDF + Cosine Similarity
# ============================================================

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. MOVIE DATASET
# ============================================================

movies = {

    "Title": [
        "Interstellar",
        "Inception",
        "The Dark Knight",
        "The Martian",
        "Avengers: Endgame",
        "Iron Man",
        "Spider-Man: No Way Home",
        "The Hangover",
        "3 Idiots",
        "Zindagi Na Milegi Dobara",
        "Dangal",
        "Taare Zameen Par",
        "Drishyam",
        "Andhadhun",
        "Stree",
        "Bhool Bhulaiyaa",
        "Dilwale Dulhania Le Jayenge",
        "Yeh Jawaani Hai Deewani",
        "Kabir Singh",
        "Jab We Met",
        "Parasite",
        "Train to Busan",
        "The Conjuring",
        "A Quiet Place",
        "Get Out",
        "The Prestige",
        "Shutter Island",
        "The Shawshank Redemption",
        "Forrest Gump",
        "The Pursuit of Happyness",
        "Avatar",
        "Dune",
        "Blade Runner 2049",
        "Arrival",
        "Ex Machina",
        "Mad Max: Fury Road",
        "John Wick",
        "Top Gun: Maverick",
        "La La Land",
        "The Notebook"
    ],

    "Genre": [
        "Sci-Fi",
        "Sci-Fi",
        "Action",
        "Sci-Fi",
        "Action",
        "Action",
        "Action",
        "Comedy",
        "Drama",
        "Drama",
        "Drama",
        "Drama",
        "Thriller",
        "Thriller",
        "Horror",
        "Horror",
        "Romance",
        "Romance",
        "Romance",
        "Romance",
        "Thriller",
        "Horror",
        "Horror",
        "Horror",
        "Horror",
        "Thriller",
        "Thriller",
        "Drama",
        "Drama",
        "Drama",
        "Sci-Fi",
        "Sci-Fi",
        "Sci-Fi",
        "Sci-Fi",
        "Sci-Fi",
        "Action",
        "Action",
        "Action",
        "Romance",
        "Romance"
    ],

    "Language": [
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "Hindi",
        "Hindi",
        "Hindi",
        "Hindi",
        "Hindi",
        "Hindi",
        "Hindi",
        "Hindi",
        "Hindi",
        "Hindi",
        "Hindi",
        "Hindi",
        "Korean",
        "Korean",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English",
        "English"
    ],

    "Mood": [
        "Emotional",
        "Suspenseful",
        "Exciting",
        "Exciting",
        "Exciting",
        "Exciting",
        "Exciting",
        "Fun",
        "Emotional",
        "Fun",
        "Emotional",
        "Emotional",
        "Suspenseful",
        "Suspenseful",
        "Fun",
        "Suspenseful",
        "Romantic",
        "Romantic",
        "Emotional",
        "Romantic",
        "Suspenseful",
        "Suspenseful",
        "Scary",
        "Suspenseful",
        "Suspenseful",
        "Suspenseful",
        "Suspenseful",
        "Emotional",
        "Emotional",
        "Emotional",
        "Exciting",
        "Exciting",
        "Suspenseful",
        "Emotional",
        "Suspenseful",
        "Exciting",
        "Exciting",
        "Exciting",
        "Romantic",
        "Romantic"
    ],

    "Rating": [
        8.7, 8.8, 9.0, 8.0, 8.4,
        7.9, 7.8, 7.7, 8.4, 8.2,
        8.3, 8.3, 8.2, 8.2, 7.3,
        7.1, 8.0, 7.2, 6.9, 7.9,
        8.5, 7.6, 7.5, 7.5, 7.7,
        8.5, 8.2, 9.3, 8.8, 8.0,
        7.8, 8.0, 8.1, 7.9, 7.7,
        8.1, 7.6, 8.3, 8.0, 7.8
    ],

    "Year": [
        2014, 2010, 2008, 2015, 2019,
        2008, 2021, 2009, 2009, 2011,
        2016, 2007, 2015, 2018, 2018,
        2007, 1995, 2013, 2019, 2007,
        2019, 2016, 2013, 2018, 2017,
        2006, 2010, 1994, 1994, 2006,
        2009, 2021, 2017, 2016, 2014,
        2015, 2014, 2022, 2016, 2004
    ],

    "Duration": [
        169, 148, 152, 144, 181,
        126, 148, 100, 170, 155,
        161, 165, 163, 139, 128,
        159, 189, 160, 173, 138,
        132, 118, 112, 90, 104,
        130, 138, 142, 142, 117,
        162, 155, 164, 116, 108,
        120, 101, 131, 128, 123
    ]
}

df = pd.DataFrame(movies)


# ============================================================
# 2. CREATE CONTENT INFORMATION
# ============================================================

# Combine important movie features into one text column

df["Content"] = (
    df["Genre"] + " "
    + df["Language"] + " "
    + df["Mood"]
)


# ============================================================
# 3. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    df["Content"]
)


# ============================================================
# 4. COSINE SIMILARITY
# ============================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# ============================================================
# 5. DISPLAY OPTIONS
# ============================================================

print("\n")
print("=" * 60)
print("             MOVIE RECOMMENDATION ENGINE")
print("=" * 60)

print("\nAnswer the following questions.\n")


# ============================================================
# QUESTION 1 - GENRE
# ============================================================

genres = [
    "Action",
    "Comedy",
    "Romance",
    "Thriller",
    "Horror",
    "Sci-Fi",
    "Drama"
]

print("1. What type of movie do you like?")

for i, genre in enumerate(genres, 1):
    print(f"{i}. {genre}")

genre_choice = int(
    input("Enter your choice: ")
)

selected_genre = genres[
    genre_choice - 1
]


# ============================================================
# QUESTION 2 - LANGUAGE
# ============================================================

languages = [
    "English",
    "Hindi",
    "Korean",
    "Any"
]

print("\n2. Which language do you prefer?")

for i, language in enumerate(languages, 1):
    print(f"{i}. {language}")

language_choice = int(
    input("Enter your choice: ")
)

selected_language = languages[
    language_choice - 1
]


# ============================================================
# QUESTION 3 - MOOD
# ============================================================

moods = [
    "Fun",
    "Emotional",
    "Suspenseful",
    "Exciting",
    "Romantic",
    "Scary"
]

print("\n3. What mood are you looking for?")

for i, mood in enumerate(moods, 1):
    print(f"{i}. {mood}")

mood_choice = int(
    input("Enter your choice: ")
)

selected_mood = moods[
    mood_choice - 1
]


# ============================================================
# QUESTION 4 - MINIMUM RATING
# ============================================================

print("\n4. What minimum IMDb-style rating do you prefer?")

selected_rating = float(
    input("Enter minimum rating (0-10): ")
)


# ============================================================
# QUESTION 5 - MOVIE ERA
# ============================================================

eras = [
    "New",
    "Old",
    "Any"
]

print("\n5. Which movie era do you prefer?")

for i, era in enumerate(eras, 1):
    print(f"{i}. {era}")

era_choice = int(
    input("Enter your choice: ")
)

selected_era = eras[
    era_choice - 1
]


# ============================================================
# QUESTION 6 - DURATION
# ============================================================

durations = [
    "Short",
    "Medium",
    "Long",
    "Any"
]

print("\n6. What movie duration do you prefer?")

for i, duration in enumerate(durations, 1):
    print(f"{i}. {duration}")

duration_choice = int(
    input("Enter your choice: ")
)

selected_duration = durations[
    duration_choice - 1
]


# ============================================================
# 6. CALCULATE RECOMMENDATION SCORE
# ============================================================

scores = []


for i, movie in df.iterrows():

    score = 0


    # --------------------------------------------------------
    # Genre match
    # --------------------------------------------------------

    if movie["Genre"] == selected_genre:
        score += 35


    # --------------------------------------------------------
    # Language match
    # --------------------------------------------------------

    if (
        selected_language == "Any"
        or movie["Language"] == selected_language
    ):
        score += 20


    # --------------------------------------------------------
    # Mood match
    # --------------------------------------------------------

    if movie["Mood"] == selected_mood:
        score += 20


    # --------------------------------------------------------
    # Rating preference
    # --------------------------------------------------------

    if movie["Rating"] >= selected_rating:
        score += 10

    else:

        # Small penalty for movies below
        # user's preferred rating

        difference = (
            selected_rating - movie["Rating"]
        )

        score -= difference * 5


    # --------------------------------------------------------
    # Era preference
    # --------------------------------------------------------

    if selected_era == "New":

        if movie["Year"] >= 2015:
            score += 10

    elif selected_era == "Old":

        if movie["Year"] < 2015:
            score += 10

    else:
        score += 10


    # --------------------------------------------------------
    # Duration preference
    # --------------------------------------------------------

    if selected_duration == "Short":

        if movie["Duration"] < 120:
            score += 5

    elif selected_duration == "Medium":

        if 120 <= movie["Duration"] <= 150:
            score += 5

    elif selected_duration == "Long":

        if movie["Duration"] > 150:
            score += 5

    else:
        score += 5


    scores.append(score)


# Add scores to dataset

df["Recommendation_Score"] = scores


# ============================================================
# 7. GET TOP 5 RECOMMENDATIONS
# ============================================================

recommendations = df.sort_values(
    by="Recommendation_Score",
    ascending=False
).head(5)


# ============================================================
# 8. DISPLAY USER PREFERENCES
# ============================================================

print("\n")
print("=" * 60)
print("                YOUR PREFERENCES")
print("=" * 60)

print("Genre          :", selected_genre)
print("Language       :", selected_language)
print("Mood           :", selected_mood)
print("Minimum Rating :", selected_rating)
print("Era            :", selected_era)
print("Duration       :", selected_duration)


# ============================================================
# 9. DISPLAY TOP 5 MOVIES
# ============================================================

print("\n")
print("=" * 60)
print("               TOP 5 RECOMMENDATIONS")
print("=" * 60)

for position, (_, movie) in enumerate(
    recommendations.iterrows(),
    1
):

    print(f"\n{position}. {movie['Title']}")

    print(
        f"   Genre       : {movie['Genre']}"
    )

    print(
        f"   Language    : {movie['Language']}"
    )

    print(
        f"   Mood        : {movie['Mood']}"
    )

    print(
        f"   Rating      : {movie['Rating']}"
    )

    print(
        f"   Year        : {movie['Year']}"
    )

    print(
        f"   Duration    : {movie['Duration']} minutes"
    )

    print(
        f"   Match Score : "
        f"{movie['Recommendation_Score']:.1f}"
    )


# ============================================================
# 10. END
# ============================================================

print("\n")
print("=" * 60)
print("        THANK YOU FOR USING THE SYSTEM!")
print("=" * 60)