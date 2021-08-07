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

# Count words occurrences
def count_words(data, dictionary, with_freq=False):
    counter = defaultdict(lambda: 0)
    for d in data:
        for w in dictionary:
            text = d["text"]
            count = text.count(w)
            if not with_freq:
                counter[d["year"]] += count
            else:
                counter[d["year"]] += abs(math.log(count / len(set(text)))) if count != 0 else 0
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
# Log probabilities

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
    text = [[token.lemma_.lower() for token in nlp(s)] for s in sentences]
    training_data_per_year[d["year"]].extend(text)
    training_data.extend(text)

model = Word2Vec(training_data, sg=1)

def get_most_similar(dictionary, model):
    most_similar = {}
    for word in dictionary:
        try:
            similars = model.wv.most_similar(positive=word)
            most_similar[word] = similars[0][0]
        except KeyError:
            pass

    return most_similar

woman_most_similar = get_most_similar(woman_dict, model)
man_most_similar = get_most_similar(man_dict, model)

# Named Entity Recognition
names = list({ent.text for d in data for ent in nlp(d["text"]).ents if ent.label_ == "PERSON"})

for i, name in enumerate(names):
    if " " in name:
        words = name.split()
        names[i] = words[0]

df = pd.read_csv("NationalNames.csv").drop(["Id", "Year", "Count"], axis=1)

x = df["Name"]
cv = CountVectorizer().fit(x)

gender_model = pickle.load(open("Logistic Regression.sav", 'rb'))

prediction = gender_model.predict(cv.transform(names).toarray())

male_names = [names[i].lower() for i, p in enumerate(prediction) if p == "M"]
female_names = [names[i].lower() for i, p in enumerate(prediction) if p == "F"]

female_names_most_similar = get_most_similar(female_names, model)
male_names_most_similar = get_most_similar(male_names, model)

print(woman_most_similar, '\n')
print(man_most_similar, '\n')
print(female_names_most_similar, '\n')
print(male_names_most_similar, '\n')

# create one model for each year
for k, v in training_data_per_year.items():
    model = Word2Vec(v, sg=1)
    woman_most_similar = get_most_similar(woman_dict, model)
    man_most_similar = get_most_similar(man_dict, model)
    print("For the year {} the woman dict is {} and the man dict is {}".format(k, woman_most_similar, man_most_similar), '\n')


















