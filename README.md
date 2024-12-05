# Movie Recommendation System

This project is a content-based movie recommendation system built using Python. By analyzing movie data such as genres, keywords, cast, and crew, the system suggests similar movies based on a given input. It uses natural language processing techniques and cosine similarity to calculate the relevance between movies.

---

## Features

- **Content-Based Filtering**: Recommends movies based on metadata such as genres, cast, crew, and keywords.
- **Efficient Data Preprocessing**: Cleans and processes large movie datasets using pandas and natural language processing.
- **Customizable Recommendations**: Allows users to input a movie name and receive a list of the top 5 most similar movies.
- **Pickle Serialization**: Saves preprocessed data and similarity scores for quick access and reuse.

---

## Prerequisites

Ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `nltk`
- `scikit-learn`

Install them using:
```bash
pip install numpy pandas nltk scikit-learn
```

---

## Project Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Add the datasets `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` to the project folder.

3. Run the script:
   ```bash
   python main.py
   ```

---

## How It Works

### Data Preprocessing
1. **Merge and Clean Datasets**:
   - Combine `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` on the title column.
   - Retain essential columns: `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, and `crew`.

2. **Text Transformation**:
   - Convert genres, keywords, cast, and crew into a list of keywords.
   - Remove spaces and lowercase all text for uniformity.

3. **Stemming**:
   - Use the PorterStemmer to reduce words to their root forms (e.g., "dancing" → "dance").

4. **Tag Creation**:
   - Combine `overview`, `genres`, `keywords`, `cast`, and `crew` into a single `tag` column for each movie.

---

### Vectorization and Similarity
1. **Count Vectorizer**:
   - Convert the `tag` column into vectors using `CountVectorizer` with a maximum of 5000 features and English stop words.

2. **Cosine Similarity**:
   - Calculate similarity scores between movies using the cosine similarity metric.

---

### Recommendation Function
The `recommend()` function takes a movie title as input and returns the top 5 most similar movies based on the computed similarity scores.

```python
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```

Example:
```python
recommend('Avatar')
# Output:
# 1. Aliens
# 2. Guardians of the Galaxy
# 3. Star Trek Into Darkness
# 4. Independence Day
# 5. Star Wars
```

---

## Saved Models and Data
- **`movies.pkl`**: Contains the processed DataFrame.
- **`movie_dict.pkl`**: Contains the dictionary form of the DataFrame for API or frontend integration.
- **`similarity.pkl`**: Contains the precomputed cosine similarity matrix for fast recommendations.

---

## Future Enhancements
- Integrate an API or web interface for user interaction (e.g., Flask or Django).
- Add user-based collaborative filtering to enhance recommendations.
- Use advanced NLP techniques (e.g., TF-IDF or BERT) for better text analysis.
