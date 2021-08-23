from model import Model
from pymongo import MongoClient

# Get data from mongodb
client = MongoClient()
db = client["vatican"]
collection = db["publications"]
data = list(collection.find(sort=[("year", 1)]))

model = Model(data)
model.run()




















