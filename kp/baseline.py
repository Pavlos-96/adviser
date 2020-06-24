import ast
import sys
sys.path.insert(0, "/Users/pavlosmusenidis/Desktop/Computerlinguistik/2.Semester/SpokenDialogueSystems/adviser/adviser")

from services.service import Service, PublishSubscribe, DialogSystem
from domain_tracker import DomainTracker
from utils.domain import Domain
from utils.domain.jsonlookupdomain import JSONLookupDomain
from sklearn.metrics import f1_score
from test import *


file = "clean_domain_data.json"
file = get_data(file)
corpus = Corpus(file)
corpus.create_objects()
Hotel = JSONLookupDomain("Hotel")
Attraction = JSONLookupDomain("Attraction")
Hospital = JSONLookupDomain("Hospital")
Police = JSONLookupDomain("Police")
Restaurant = JSONLookupDomain("Restaurant")
Taxi = JSONLookupDomain("Taxi")
Train = JSONLookupDomain("Train")


dt = DomainTracker([Hotel, Attraction, Hospital, Police, Restaurant, Taxi, Train])

for dialogue in corpus.processed_corpus:
    dt.dialog_start()
    for sentence in dialogue.sentences:
        try:
            sentence.pred_domain = [dt.select_domain(sentence.string)["predicted domain: "]]
        except:
            sentence.pred_domain = []
