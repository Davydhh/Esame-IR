from model import Model
from pymongo import MongoClient

# Get data from mongodb
client = MongoClient()
db = client["vatican"]
collection = db["publications"]
data = list(collection.find(sort=[("year", 1)]))

model = Model(data)
model.run()

pipeline = [
    {
        "$group": {
            "_id": "$year",
            "documents": {
                "$addToSet": {
                    "_id": "$_id",
                    "text": "$text",
                    "pope": "$pope",
                    "type": "$type",
                    "year": "$year"
                }
            }
        }
    },
    {
        "$sort": {
            "_id": 1
        }
    }
]

data = collection.aggregate(pipeline)

for d in data:
    print("Year {}".format(d["_id"]))
    data = list(collection.find(sort=[("year", 1)]))
    model = Model(d["documents"])
    model.word_embeddings()