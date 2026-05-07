import streamlit as st
import pandas as pd
import nltk
import string
import numpy as np
import io
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===============================
# DOWNLOAD NLP RESOURCES
# ===============================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("recipes_clean.csv", encoding='utf-8')
df.columns = df.columns.str.strip()
df = df.fillna("")


# ===============================
# PREPROCESSING
# ===============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

synonyms = {
    "aloo": "potato",
    "potatoes": "potato",
    "onions": "onion",
    "tomatoes": "tomato",
    "capsicum": "bellpepper",
    "chana": "chickpea"
}

def preprocess(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [synonyms.get(w, w) for w in words]
    return " ".join(words)


# ===============================
# TF-IDF MODELS
# ===============================
df["combined_text"] = (
    df["name_of_dish"] + " " +
    df["ingredients"] + " " +
    df["course"] + " " +
    df["cuisine_state"]
)

df["cleaned_text"]        = df["combined_text"].apply(preprocess)
df["cleaned_ingredients"] = df["ingredients"].apply(preprocess)
df["cleaned_flavour"]     = df["flavour_type"].apply(preprocess)

recipe_vectorizer     = TfidfVectorizer()
ingredient_vectorizer = TfidfVectorizer()
flavour_vectorizer    = TfidfVectorizer()

recipe_X     = recipe_vectorizer.fit_transform(df["cleaned_text"])
ingredient_X = ingredient_vectorizer.fit_transform(df["cleaned_ingredients"])
flavour_X    = flavour_vectorizer.fit_transform(df["cleaned_flavour"])


# ===============================
# SEARCH FUNCTIONS
# ===============================
def search_recipe(query, top_n=5):
    query = preprocess(query)
    vec = recipe_vectorizer.transform([query])
    sim = cosine_similarity(vec, recipe_X)[0]
    top_idx = sim.argsort()[::-1][:top_n]
    results = []
    for i in top_idx:
        if sim[i] > 0:
            results.append(df.iloc[i]["name_of_dish"])
    return results


def search_by_ingredients(text):
    text = text.lower().replace("i have", "").replace("with", "").replace("and", "")
    text = preprocess(text)
    vec = ingredient_vectorizer.transform([text])
    sim = cosine_similarity(vec, ingredient_X)[0]
    top_idx = sim.argsort()[::-1][:5]
    results = []
    for i in top_idx:
        if sim[i] > 0:
            dish = str(df.iloc[i]["name_of_dish"]).strip()
            if dish and dish.lower() != "nan":
                results.append(dish)
    return results


def search_by_flavour(text):
    # Strip trigger phrases so only the flavour keywords remain
    text = text.lower()
    for phrase in ["i want something", "i want", "something", "craving", "mood for", "i feel like", "show me"]:
        text = text.replace(phrase, "")
    text = preprocess(text.strip())
    vec = flavour_vectorizer.transform([text])
    sim = cosine_similarity(vec, flavour_X)[0]
    top_idx = sim.argsort()[::-1][:5]
    results = []
    for i in top_idx:
        if sim[i] > 0:
            dish = str(df.iloc[i]["name_of_dish"]).strip()
            if dish and dish.lower() != "nan":
                results.append(dish)
    return results


def is_flavour_query(text):
    """Detect if the user is searching by flavour/mood."""
    flavour_keywords = [
        "sweet", "spicy", "tangy", "creamy", "crispy", "smoky", "savory",
        "mild", "rich", "earthy", "sour", "bitter", "aromatic", "fresh",
        "crunchy", "nutty", "hearty", "light", "pungent", "cooling",
        "warm", "fragrant", "i want something", "craving", "mood for",
        "i feel like", "something"
    ]
    return any(kw in text for kw in flavour_keywords)


# ===============================
# STREAMLIT UI
# ===============================
st.title("🍛 Indian Veg Recipe Chatbot")
st.write("Ask me recipes, tell me ingredients you have, or describe a flavour you're craving!")

# Initialize session state
if "selected_recipe" not in st.session_state:
    st.session_state.selected_recipe = None
if "results" not in st.session_state:
    st.session_state.results = []
if "mode" not in st.session_state:
    st.session_state.mode = None

# ---- INPUT + BUTTONS ----
user_input = st.text_input("Type your query")

col1, col2 = st.columns([1, 1])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear")

if clear:
    st.session_state.selected_recipe = None
    st.session_state.results = []
    st.session_state.mode = None
    st.rerun()

if send:
    if user_input.strip() == "":
        st.warning("Please enter something!")
    else:
        st.session_state.selected_recipe = None
        text = user_input.lower()

        if "i have" in text:
            st.session_state.results = search_by_ingredients(text)
            st.session_state.mode = "ingredients"
        elif is_flavour_query(text):
            st.session_state.results = search_by_flavour(text)
            st.session_state.mode = "flavour"
        else:
            st.session_state.results = search_recipe(text)
            st.session_state.mode = "search"

# ---- SHOW RESULTS ----
if st.session_state.mode == "ingredients" and st.session_state.results:
    st.subheader("🍽 You can make:")
    for name in st.session_state.results:
        if st.button(f"{name}", key=name):
            st.session_state.selected_recipe = name

elif st.session_state.mode == "flavour" and st.session_state.results:
    st.subheader("🍽 Recipes matching your craving:")
    for name in st.session_state.results:
        if st.button(f"{name}", key=name):
            st.session_state.selected_recipe = name

elif st.session_state.mode == "search" and st.session_state.results:
    st.subheader("🍽 Best Matching Recipes:")
    for name in st.session_state.results:
        if st.button(f"{name}", key=name):
            st.session_state.selected_recipe = name

# ---- SHOW FULL RECIPE ----
if st.session_state.selected_recipe:
    match = df[df["name_of_dish"] == st.session_state.selected_recipe]
    if not match.empty:
        recipe = match.iloc[0]
        st.markdown("---")
        st.subheader(f"{recipe['name_of_dish']}")
        st.markdown(f"**Ingredients:** {recipe['ingredients']}")
        st.markdown(f"**Prep Time:** {recipe['preparation_time']}")
        st.markdown(f"**Cook Time:** {recipe['cooking_time']}")
        st.markdown(f"**Cuisine:** {recipe['cuisine_state']}")
        st.markdown(f"**Flavour:** {recipe['flavour_type']}")
        st.markdown("**Method:**")
        if str(recipe["method"]).strip() in ["", "nan", "NaN"]:
            st.info("Method not available for this recipe.")
        else:
            steps = str(recipe["method"]).split("|")
            for j, step in enumerate(steps):
                st.markdown(f"{j+1}. {step.strip()}")

# ===============================
# MODEL EVALUATION USING ROUGE
# ===============================
from rouge_score import rouge_scorer
import random

# initialize scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# number of samples
sample_size = 100  # increase to 200 if needed

indices = random.sample(range(len(df)), sample_size)
scores = []

for i in indices:
    # simulate user query (dish name)
    query = df.iloc[i]["name_of_dish"]

    # chatbot prediction (returns list of dish names)
    predicted_recipes = search_recipe(query)

    # skip if no prediction
    if len(predicted_recipes) == 0:
        continue

    # take top prediction
    top_recipe_name = predicted_recipes[0]

    # get predicted recipe full text from dataframe
    pred_match = df[df["name_of_dish"] == top_recipe_name]

    if pred_match.empty:
        continue

    predicted_text = pred_match.iloc[0]["combined_text"]

    # reference (actual recipe)
    reference_text = df.iloc[i]["combined_text"]

    # compute ROUGE
    score = scorer.score(reference_text, predicted_text)
    scores.append(score['rougeL'].fmeasure)

# avoid division error
if len(scores) > 0:
    avg_score = sum(scores) / len(scores)
else:
    avg_score = 0

print("Average ROUGE-L Score:", avg_score)
print("Evaluation complete")