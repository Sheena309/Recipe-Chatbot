# Recipe-Chatbot
# Indian Veg Recipe Chatbot

A retrieval-based chatbot that helps users discover Indian vegetarian recipes by dish name, available ingredients, or flavour preferences — built with TF-IDF and Cosine Similarity.

---

## About the Project

Built by **Sheena Munjal**

Food is something everyone relates to, yet many people struggle with deciding what to cook based on what they have at home or what they're in the mood for. This chatbot solves that by letting users:

- Search recipes by **dish name**
- Get suggestions based on **ingredients they have**
- Get recommendations based on **flavour preferences** (spicy, sweet, creamy, etc.)

---

## Dataset

**File:** `recipes_clean.csv`

The dataset contains Indian vegetarian recipes. Each row represents one recipe with the following fields:

| Column | Description |
|---|---|
| `name_of_dish` | Name of the recipe |
| `ingredients` | List of ingredients |
| `cuisine_state` | Cuisine or state of origin |
| `course` | Course type (e.g., main, snack) |
| `flavour_type` | Flavour profile (e.g., spicy, sweet) |
| `method` | Step-by-step cooking instructions |
| `preparation_time` | Prep time |
| `cooking_time` | Cook time |

> The dataset acts as the **knowledge base** for the chatbot.

---

## How It Works

### 1. Preprocessing
Before searching, all text is cleaned and normalized:
- Converted to lowercase
- Punctuation removed
- Stopwords removed (e.g., "and", "the", "is")
- Lemmatization applied (e.g., "tomatoes" → "tomato")
- Synonym mapping applied (e.g., "aloo" → "potato", "chana" → "chickpea")
- Multiple columns combined into one text field for richer search

### 2. TF-IDF Vectorization
Three separate TF-IDF models are built:
- **Recipe model** — on combined text (name + ingredients + course + cuisine)
- **Ingredient model** — on ingredients only
- **Flavour model** — on flavour type only

### 3. Cosine Similarity Search
User input is converted into a vector and compared against all recipe vectors. The most similar recipes are returned as results.

> This is a **retrieval-based chatbot**, not a generative AI model — making it fast, lightweight, and easy to interpret.

---

## Architecture Summary

```
User Query
    ↓
Preprocessing (lowercase, lemmatize, synonyms)
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity vs. Recipe Database
    ↓
Top Matching Recipes Returned
```

---

## Evaluation

The system is evaluated using **ROUGE-L score**:
- 100 random recipes are sampled from the dataset
- The chatbot's top prediction is compared to the actual recipe text
- Average ROUGE-L F-measure is computed

ROUGE-L measures the longest common subsequence between predicted and reference text — a standard metric for retrieval and summarization systems.

---

## Tech Stack

- **Python**
- **Streamlit** — UI framework
- **NLTK** — text preprocessing (stopwords, lemmatization)
- **Scikit-learn** — TF-IDF vectorization, cosine similarity
- **Pandas / NumPy** — data handling
- **rouge-score** — evaluation

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Sheena309/indian-veg-recipe-chatbot.git
cd indian-veg-recipe-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> Make sure `recipes_clean.csv` is in the same folder as `app.py`.

---

## Requirements

Create a `requirements.txt` with:

```
streamlit
pandas
numpy
nltk
scikit-learn
rouge-score
```

---

## Example Queries

| Query | Mode Triggered |
|---|---|
| `butter paneer masala` | Recipe search |
| `I have potato, onion, tomato` | Ingredient-based search |
| `I want something spicy` | Flavour-based search |
| `craving something creamy` | Flavour-based search |

---

## Project Structure

```
├── app.py               # Main chatbot application
├── recipes_clean.csv    # Recipe dataset
├── requirements.txt     # Requirements file
└── README.md            # Project documentation
```

---

## Author

**Sheena Munjal**  
[GitHub Profile](https://github.com/Sheena309)

