import spacy
import math
import matplotlib.pyplot as plt

from collections import defaultdict
from pymongo import MongoClient
from nltk.corpus import wordnet

# Get data from mongodb
client = MongoClient()
db = client["vatican"]
collection = db["publications"]
data = list(collection.find(sort=[("year", 1)]))

# Tokenize and lemmatize corpus
nlp = spacy.load("en_core_web_sm")
for d in data:
    text = d["text"]
    d["text"] = [token.lemma_.lower() for token in nlp(text)]

# Create dictionaries about man and woman words
woman_dict = ["female", "girl", "woman", "she", "sister", "mrs"]
man_dict = ["male", "boy", "man", "he", "brother", "mr"]

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
                counter[int(d["year"])] += count
            else:
                counter[int(d["year"])] += abs(math.log(count / len(set(text)))) if count != 0 else 0
    return counter

woman_occurrences = count_words(data, woman_dict)
man_occurrences = count_words(data, man_dict)

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
# Unigram with relative frequency

woman_occurrences = count_words(data, woman_dict, with_freq=True)
man_occurrences = count_words(data, man_dict, with_freq=True)

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










