import spacy
import math
import nltk
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from pymongo import MongoClient
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

# Get data from mongodb
client = MongoClient()
db = client["vatican"]
collection = db["publications"]
data = list(collection.find(sort=[("year", 1)]))

# Tokenize and lemmatize corpus
nlp = spacy.load("en_core_web_sm")
parsed_data = []
for d in data:
    text = [token.lemma_.lower() for token in nlp(d["text"])]
    parsed_data.append({"_id": d["_id"], "text": text, "pope": d["pope"], "year": int(d["year"])})

# Create dictionaries about man and woman words
woman_dict = ["female", "girl", "woman", "she", "sister", "mother", "mrs", "her", "nun", "daughter", "lady"]
man_dict = ["male", "boy", "man", "he", "brother", "father", "mr", "his", "priest", "son"]

def get_syns(dictionary):
    result = set()
    for w in dictionary:
        syns = wordnet.synsets(w)
        for s in syns:
            result.add(s.lemmas()[0].name().lower())
    return list(result)

woman_dict.extend(get_syns(woman_dict))
man_dict.extend(get_syns(man_dict))

# Count words occurrences also with Laplace smoothing
def count_words(data, dictionary, with_freq=False):
    counter = defaultdict(lambda: 0)
    for d in data:
        for w in dictionary:
            text = d["text"]
            count = text.count(w)
            if not with_freq:
                counter[d["year"]] += count
            else:
                counter[d["year"]] += math.log((count + 1) / (len(text) + len(set(text))))
    return counter

woman_occurrences = count_words(parsed_data, woman_dict)
man_occurrences = count_words(parsed_data, man_dict)

def my_div(n, d):
    return n / d if d else n

def get_ratio(woman_occurrences, man_occurrences):
    ratios = []
    for y in woman_occurrences.keys():
        woman_occ = woman_occurrences[y]
        man_occ = man_occurrences[y]
        # print("For year {} woman occurrences are {} while man occurrences are {}".format(y, woman_occ, man_occ))
        ratio = round(my_div(man_occ, woman_occ))
        # print("The ration between man and woman occurences are {}".format(ratio), '\n')
        ratios.append(ratio)
    return ratios

ratios = get_ratio(woman_occurrences, man_occurrences)

# Plot data
plt.figure(figsize=(11, 10)).tight_layout()
plt.subplot(221)
plt.plot(woman_occurrences.keys(), woman_occurrences.values(), color="red")
plt.xlabel("years")
plt.ylabel("occurrences")
plt.title("Woman")
plt.subplot(222)
plt.plot(man_occurrences.keys(), man_occurrences.values())
plt.xlabel("years")
plt.ylabel("occurrences")
plt.title("Man")
plt.subplot(223)
plt.plot(woman_occurrences.keys(), woman_occurrences.values(), color="red")
plt.plot(woman_occurrences.keys(), man_occurrences.values())
plt.xlabel("years")
plt.ylabel("occurrences")
plt.title("Man and Womand")
plt.subplot(224)
plt.plot(woman_occurrences.keys(), ratios, color="purple")
plt.xlabel("years")
plt.ylabel("ratio")
plt.title("Ratio man-woman")
plt.suptitle("Basic occurrences counter")

### Language Models
## Basics
# Log probabilities with Laplace Smoothing

woman_occurrences = count_words(parsed_data, woman_dict, with_freq=True)
man_occurrences = count_words(parsed_data, man_dict, with_freq=True)

ratios = get_ratio(woman_occurrences, man_occurrences)

# Plot data
plt.figure(figsize=(11, 10)).tight_layout()
plt.subplot(221)
plt.plot(woman_occurrences.keys(), woman_occurrences.values(), color="red")
plt.xlabel("years")
plt.ylabel("occurrences")
plt.title("Woman")
plt.subplot(222)
plt.plot(man_occurrences.keys(), man_occurrences.values())
plt.xlabel("years")
plt.ylabel("occurrences")
plt.title("Man")
plt.subplot(223)
plt.plot(woman_occurrences.keys(), woman_occurrences.values(), color="red")
plt.plot(woman_occurrences.keys(), man_occurrences.values())
plt.xlabel("years")
plt.ylabel("occurrences")
plt.title("Man and Womand")
plt.subplot(224)
plt.plot(woman_occurrences.keys(), ratios, color="purple")
plt.xlabel("years")
plt.ylabel("ratio")
plt.title("Ratio man-woman")
plt.suptitle("Occurrences with relative frequency")

# plt.show()

# ## Word Embeddings
# # Word2Vec

training_data_per_year = defaultdict(lambda: [])
training_data = []
for d in data:
    sentences = nltk.tokenize.sent_tokenize(d["text"])
    text = [[token.lemma_ for token in nlp(s) if token.lemma_.isalpha()] for s in sentences]
    training_data_per_year[d["year"]].extend(text)
    training_data.extend(text)

model = Word2Vec(training_data, sg=1)

def get_most_similar(dictionary, model):
    most_similar = {}
    for word in dictionary:
        try:
            most_similar[word] = [w[0] for w in model.wv.most_similar(positive=word, topn=3)]
        except KeyError:
            pass

    return most_similar

woman_most_similar = get_most_similar(woman_dict, model)
man_most_similar = get_most_similar(man_dict, model)

woman_df = pd.DataFrame.from_dict(woman_most_similar, orient="index")
man_df = pd.DataFrame.from_dict(man_most_similar, orient="index")

# Visualize Word Embeddings
X = model.wv.vectors
pca = PCA(n_components=2)
result = pca.fit_transform(X)
words = list(model.wv.index_to_key)

def scatter_words(model, dictionary, result, words, suptitle, title):
    for k, v in dictionary.items():
        if v[0] in words:
            index = model.wv.get_index(v[0])
            plt.scatter(result[index, 0], result[index, 1], c="b", marker=',')
            plt.annotate(v[0], xy=(result[index, 0], result[index, 1]))
        if k in words:
            index = model.wv.get_index(k)
            plt.scatter(result[index, 0], result[index, 1], s=80, c='r')
            plt.annotate(k, xy=(result[index, 0], result[index, 1]))
    plt.suptitle(suptitle)
    plt.title(title)

plt.figure(figsize=(20, 10)).tight_layout()
plt.subplot(121)
scatter_words(model, woman_most_similar, result, words, "Word Embedding representation", "Woman words")
plt.subplot(122)
scatter_words(model, man_most_similar, result, words, "Word Embedding representation", "Man words")
    
plt.show()

# Named Entity Recognition
names = list({ent.text for d in data for ent in nlp(d["text"]).ents if ent.label_ == "PERSON"})

for i, name in enumerate(names):
    if " " in name:
        words = name.split()
        names[i] = words[0]

df = pd.read_csv("NationalNames.csv").drop(["Id", "Year", "Count"], axis=1)

x = df["Name"]
cv = CountVectorizer().fit(x)

gender_model = pickle.load(open("Multinomial Naive Bayes.sav", 'rb'))

prediction = gender_model.predict(cv.transform(names).toarray())

male_names = [names[i] for i, p in enumerate(prediction) if p == "M"]
female_names = [names[i] for i, p in enumerate(prediction) if p == "F"]

female_names_most_similar = get_most_similar(female_names, model)
male_names_most_similar = get_most_similar(male_names, model)

female_name_df = pd.DataFrame.from_dict(female_names_most_similar, orient="index")
male_names_df = pd.DataFrame.from_dict(male_names_most_similar, orient="index")

print(woman_df, '\n')
print(man_df, '\n')
print(female_name_df, '\n')
print(male_names_df, '\n')

# create one model for each year
# for k, v in training_data_per_year.items():
#     model = Word2Vec(v, sg=1)
#     woman_most_similar = get_most_similar(woman_dict, model)
#     man_most_similar = get_most_similar(man_dict, model)
#     print("For the year {} the woman dict is {} and the man dict is {}".format(k, woman_most_similar, man_most_similar), '\n')


















