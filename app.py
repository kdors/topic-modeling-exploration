import streamlit as st
import pandas as pd
import numpy as np
from scipy import linalg
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition


st.title("Explore Topic Modeling with Bob's Burgers :hamburger:")

st.markdown("""
### Some Background

I have recently started studying natural language processing, and more specifically, topic modeling. I find the idea of generating topics
from a collection of topics fascinating, and have been playing around with different methods to generate said topics in Jupyter notebooks.
Becasuse of this, I decided it would be cool to put it all together in a Streamlit app to see it work in real-time. It also gives me an 
excuse to try out Streamlit, which has been really fun!

As I get more experience with topic modeling, I hope to add more to this app!

### Why Bob's Burgers

Bob's Burgers is one of my favorite TV shows to watch when I want to relax and enjoy wholesome family TV.

If you have never watched the show, it's about all of the crazy things that happen to a middle-class family of size five (parents Bob and 
Linda, children Tina, Gene, and Louise) that own a burger place. You can check out the [wiki](https://en.wikipedia.org/wiki/Bob%27s_Burgers) 
that will go much more in depth if you're interested (Or you can always just watch an episode or two!). 

### Let's do some topic modeling

The data used are the scripts of Bob's Burgers episodes from seasons 1-5. You can see the scraping and cleaning process in this 
[notebook.](https://github.com/kdors/topic-modeling-exploration/blob/main/topicmodeling.ipynb) I hope to add more seasons in the future!

### Term Frequency

There are different methods for tokenization and for deciding how to count the tokens/terms. In this app, you can use CountVectorizer or
TfidfVectorizer. CountVectorizer counts the tokens in each document, while TfidfVectorizer also normalizes them so that super
common words aren't weighted too heavily.

You can also decide whether or not you want to exclude stop words or not. Stop words are common words that appear in most of the documents 
used in the modeling process. If you want to exlcude stop words, you can choose to use Scikit-learn's stop words list.
""")

@st.cache
def get_data():
    data = []
    with open("bb_scripts_file.txt", "r", encoding="UTF-8") as inpt:
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


add_stopwords = st.sidebar.selectbox(
    "Stop Words",
    ("No stop words", "Scikit-learn")
)

add_vectorizer = st.sidebar.selectbox(
    "Term Vectorization",
    ("CountVectorizer", "TFIDFVectorizer")
)

add_factorization = st.sidebar.selectbox(
    "Matrix Factorization",
    ("SVD", "NMF")
)

add_topic_num = st.sidebar.slider(
    "Number of topics", 5, 40, 8)

add_words_num = st.sidebar.slider(
    "Number of words per topic", 5, 20, 8)

num_top_words = add_words_num
@st.cache
def show_topics(arr):
    top_words = lambda t: [vocab_words[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in arr])
    return [' '.join(t) for t in topic_words]


if add_vectorizer == "CountVectorizer":
    if add_stopwords == "No stop words":
        vect = CountVectorizer()
    else:
        vect = CountVectorizer(stop_words="english")
else:
    if add_stopwords == "No stop words":
        vect = TfidfVectorizer()
    else:
        vect = TfidfVectorizer(stop_words="english")


data_tokenized = vect.fit_transform(scripts).todense()
vocab_words = np.array(vect.get_feature_names_out())

st.metric(label="Number of episodes", value=len(scripts))

st.metric(label="Vocabulary Length", value=len(vocab_words))

st.write("A sneak peek at some vocabulary words after tokenization:")

st.write(vocab_words[100::1000])

st.write("After using CountVectorizer/TfidfVectorizer, the resulting vector shape is:", data_tokenized.shape, 
        "as expected (number of episodes x number of terms).")


st.markdown(
    """
    ### Matrix Factorization

    The two options in this app for matrix factorization are SVD (singular value decomposition) and NMF (nonnegative matrix 
    factorization). SVD produces an exact decompositon of the data matrix, while NMF produces a non-exact decomposition. 

    SVD can be really show with a large document-term matrix, but since we only have 88 episodes here, it's decently fast.
    """
)

st.markdown(
    """
    ### Topics Created
    """
)


if add_factorization == "SVD":
    episode_topic, S, V_t = linalg.svd(data_tokenized, full_matrices=False)
    st.write(show_topics(V_t[:add_topic_num]))  

else:
    nmf_decomp = decomposition.NMF(n_components=add_topic_num, random_state=1)
    W1 = nmf_decomp.fit_transform(data_tokenized)
    H1 = nmf_decomp.components_
    st.write(show_topics(H1))

if add_factorization == "SVD":
    episode_topic = episode_topic
else:
    episode_topic = W1

st.markdown(
    """
    ### How well does it work?

    Is there some configuration that produces good results?

    If I pick a Christmas episode, say **4x08 "Christmas in a Car",** are there any topics above that seem Christmas-y, 
    and does the episode correspond the most with that topic? (If you don't see a topic with any Christmas-adjacent words, 
    try adding more topics!).

    Let's find out. The indices of the table below match the topics generated above and the numbers next to them correspond to
    how well each topic matches episode 4x08, with the higher number meaning the more you can find that topic in the episode script.
    """
)

st.write(episode_topic[52,:add_topic_num])

episode = st.selectbox("Try it out with different episodes!", 
            ("1x06 Sheesh! Cab, Bob?","2x08 Bad Tina", "3x12 Broadcast Wagstaff School News",
            "3x13 My Fuzzy Valentine", "4x02 Full Bars (Halloween episode)", "4x05 Turkey in a Can (Thanksgiving episode)", 
            "4x20 Gene It On", "5x04 Dawn of the Peck (Thanksgiving episode)", "5x06 Father of the Bob (Christmas episode)",
            "5x10 Late Afternoon in the Garden of Bob and Louise"))


if episode[:4] == "1x06":
    st.write(episode_topic[5,:add_topic_num])
elif episode[:4] == "2x08":
    st.write(episode_topic[20,:add_topic_num])
elif episode[:4] == "3x12":
    st.write(episode_topic[33,:add_topic_num])
elif episode[:4] == "3x13":
    st.write(episode_topic[34,:add_topic_num])
elif episode[:4] == "4x02":
    st.write(episode_topic[46,:add_topic_num])
elif episode[:4] == "4x05":
    st.write(episode_topic[49,:add_topic_num])
elif episode[:4] == "4x20":
    st.write(episode_topic[64,:add_topic_num])
elif episode[:4] == "5x04":
    st.write(episode_topic[70,:add_topic_num])
elif episode[:4] == "5x06":
    st.write(episode_topic[72,:add_topic_num])
else:
    st.write(episode_topic[76,:add_topic_num])
