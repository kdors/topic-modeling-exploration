import streamlit as st
import pandas as pd
import numpy as np
from scipy import linalg
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition


st.title("Exploring Topic Modeling with Bob's Burgers :hamburger:")

st.markdown("""
### Some Background

I have recently started studying natural language processing, and more specifically, topic modeling. I find the idea of generating topics
from a collection of topics fascinating, and have been playing around with different methods to generate said topics in Jupyter notebooks.
Becasuse of this, I decided it would be cool to put it all together in a Streamlit app to see it work in real-time. It also gives me an 
excuse to try out Streamlit, which has been really fun!

As I get more experience with topic modeling, I hope to add more to this app!

### Why Bob's Burgers

Bob's Burgers is one of my favorite TV shows to watch when I want to relax and enjoy wholesome family TV.

If you have never watched the show, it's about all of the crazy things that happen to a middle-class family of size five that own a burger 
place. You check out the [wiki](https://en.wikipedia.org/wiki/Bob%27s_Burgers) that will go much more in depth if you're interested 
(Or you can always just watch an episode or two!). 

### Let's do some topic modeling

The data used are the scripts of Bob's Burgers episodes from seasons 1-3. You can see the scraping and cleaning process in this 
[notebook.](https://github.com/kdors/topic-modeling-exploration/blob/main/topicmodeling.ipynb) I hope to add more seasons in the future!

### Term Frequency

There are different methods for tokenization and for deciding how to count the tokens/terms. In this app, you can use CountVectorizer or
TfidfVectorizer. CountVectorizer simply counts the tokens in each document, while TfidfVectorizer also normalizes them so that super
common words aren't weighted too heavily.

You can also decide whether or not you want to exclude stop words or not. Stop words are common words that appear in most of the documents 
used in the modeling process. If you want to exlcude stop words, you can choose to use Scikit-learn's stop words list.
""")

@st.cache
def get_data():
    data = []
    with open("scripts_file.txt", "r", encoding="UTF-8") as inpt:
        for episode in inpt:
            data.append(episode) 
        
    return data


scripts = get_data()

st.sidebar.markdown(
    """
    ### Topic Modeling Configuation
    Hey! :wave: This is an app to explore topic modeling in real time and see how different processes can change the topics generated.

    **Different configurations:**
    """
)

add_selectbox_stopwords = st.sidebar.selectbox(
    "Stop Words",
    ("No stop words", "Scikit-learn")
)

add_selectbox_tokenizer = st.sidebar.selectbox(
    "Term Vectorization",
    ("CountVectorizer", "TFIDFVectorizer")
)

add_selectbox_factorization = st.sidebar.selectbox(
    "Matrix Factorization",
    ("SVD", "NMF")
)

add_selectbox_topics = st.sidebar.selectbox(
    "Number of topics",
    (5, 10, 20, 30, 40)
)

add_selectbox_topic_words = st.sidebar.selectbox(
    "Number of words per topic",
    (5, 8, 10)
)


if add_selectbox_tokenizer == "CountVectorizer":
    if add_selectbox_stopwords == "No stop words":
        vect = CountVectorizer()
    else:
        vect = CountVectorizer(stop_words="english")
else:
    if add_selectbox_stopwords == "No stop words":
        vect = TfidfVectorizer()
    else:
        vect = TfidfVectorizer(stop_words="english")

data_tokenized = vect.fit_transform(scripts).todense()
vocab_words = np.array(vect.get_feature_names_out())

st.write("Number of Documents: ", len(scripts))
st.write("Vocabulary size: ", len(vocab_words))

st.write("A sneak peek at some vocabulary words after term vectorization:")

word_counts = [vect.vocabulary_[val] for val in vocab_words[100::700]]
df = pd.DataFrame({"Term":vocab_words[100::700], "Count":word_counts})
st.table(df)


st.markdown(
    """
    ### Matrix Factorization

    The two options in this app for matrix facorzation are SVD (singular value decomposition) and NMF (nonnegative matrix 
    factorization). SVD produces an exact decompositon of the data matrix, while NMF is non-exact.
    """
)

st.markdown(
    """
    ### Topics Created
    """
)

num_top_words = add_selectbox_topic_words
def show_topics(arr):
    top_words = lambda t: [vocab_words[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in arr])
    return [' '.join(t) for t in topic_words]

if add_selectbox_factorization == "SVD":
    U, S, V_t = linalg.svd(data_tokenized, full_matrices=False)
    st.write(show_topics(V_t[:add_selectbox_topics]))  

else:
    nmf_decomp = decomposition.NMF(n_components=add_selectbox_topics, random_state=1)
    W1 = nmf_decomp.fit_transform(data_tokenized)
    H1 = nmf_decomp.components_
    st.write(show_topics(H1))