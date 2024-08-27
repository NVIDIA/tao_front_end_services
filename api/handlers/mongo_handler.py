import pymongo
import os
from urllib import parse

mongo_secret = os.getenv("MONGOSECRET")
encoded_secret = parse.quote(mongo_secret, safe='')
mongo_connection_string = f"mongodb+srv://default-user:{encoded_secret}@mongodb-svc.default.svc.cluster.local/tao?replicaSet=mongodb&ssl=false&authSource=admin"
mongo_client = pymongo.MongoClient(mongo_connection_string, tz_aware=True)


class MongoHandler:

    def __init__(self, db_name, collection_name):
        self.mongo_client = mongo_client
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]
        self.create_unique_index("id")

    def upsert(self, query, new_data):
        self.collection.update_one(query, {'$set': new_data}, upsert=True)

    def delete_one(self, query):
        self.collection.delete_one(query)

    def delete_many(self, query):
        self.collection.delete_many(query)

    def find(self, query):
        if not query:
            result = list(self.collection.find())
        else:
            result = list(self.collection.find(query))

        return result if result else []

    def find_one(self, query=None):
        if not query:
            result = self.collection.find_one()
        else:
            result = self.collection.find_one(query)

        return result if result else {}

    def create_unique_index(self, index):
        self.collection.create_index(index, unique=True)

    def create_text_index(self, index):
        self.collection.create_index([(index, pymongo.TEXT)])

    def create_ttl_index(self, index, ttl_time_in_seconds):
        self.collection.create_index(index, expireAfterSeconds=ttl_time_in_seconds)
