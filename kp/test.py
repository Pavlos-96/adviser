import ast
import sys
sys.path.insert(0, "/Users/pavlosmusenidis/Desktop/Computerlinguistik/2.Semester/SpokenDialogueSystems/adviser/adviser")

from services.service import Service, PublishSubscribe, DialogSystem
from domain_tracker import DomainTracker
from utils.domain import Domain
from utils.domain.jsonlookupdomain import JSONLookupDomain
from sklearn.metrics import f1_score

def get_data(file):
    f = open(file, "r")
    contents = f.read()
    dictionary = ast.literal_eval(contents)
    f.close()
    return dictionary

class Sentence:
    def __init__(self, string, pred_domain="", gold_domain=""):
        self.string = string # "I want wifi"
        self.pred_domain = pred_domain # "Train"
        self.gold_domain = gold_domain # "Hotel"


class Dialogue:
    def __init__(self, sentences=None):
        if sentences is None:
            sentences = list()
        self.sentences = sentences


class Corpus:  # input is dictionary with data
    def __init__(self, dictionary, all_tags=None, processed_corpus=None):
        self.dictionary = dictionary
        if all_tags is None:
            all_tags = set()
        self.all_tags = all_tags  # set of all tags in gold + predicted
        if processed_corpus is None:
            processed_corpus = list()
        self.processed_corpus = processed_corpus

    def create_objects(self):  # make sentence + token objects, no input
        for data in self.dictionary:  # e.g ['We', 'are', 'happy', '.']
            sentences = []  # list of sentence objects
            for evaluation in self.dictionary[data]:  # sentence
                sentence_obj = Sentence(evaluation[0])  # creates tokenobj
                sentence_obj.gold_domain = evaluation[1]
                if len(sentence_obj.gold_domain) != 0:
                    for tag in sentence_obj.gold_domain:
                        self.all_tags.update({tag})
                sentences.append(sentence_obj)
            dialogue_obj = Dialogue(sentences)  # creates sentence object
            self.processed_corpus.append(dialogue_obj)


class Comparison:
    # objects of this class contain results of the evaluation of specific POS tags
    def __init__(self, domain,
                 comparison=None,
                 precision=0, recall=0, accuracy=0, fscore=0):
        self.domain = domain # the identity of the object
        if comparison is None:
            comparison = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        self.comparison = comparison
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.fscore = fscore

#[Train{'TP': 99, 'FP': 34, 'FN': 0, 'TN': 0},Hotel{'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},Attraction{'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    def get_comparison(self, corpus):
        # for each domain find out the TPs FPs and FNs and save them in the self.comparison dictionary
        for dialogue in corpus.processed_corpus:
            for sentence in dialogue.sentences:
                if self.domain in sentence.gold_domain and self.domain in sentence.pred_domain:  # TP
                    self.comparison['TP'] += 1
                elif self.domain not in sentence.gold_domain and self.domain in sentence.pred_domain:  # FP
                    self.comparison['FP'] += 1
                elif self.domain in sentence.gold_domain and self.domain not in sentence.pred_domain:  # FN
                    self.comparison['FN'] += 1
                elif self.domain not in sentence.gold_domain and self.domain not in sentence.pred_domain:  #TN
                    self.comparison['TN'] += 1

    def get_precision(self):
        self.precision = self.comparison['TP']/(self.comparison['TP'] + self.comparison['FP'])

    def get_recall(self):
        self.recall = self.comparison['TP']/(self.comparison['TP'] + self.comparison['FN'])

    def get_accuracy(self):
        self.accuracy = (self.comparison['TP']+self.comparison['TN'])/\
                        (self.comparison['TP'] +self.comparison['TN'] + self.comparison['FP'] + self.comparison['FN'])

    def get_fscore(self):
        self.fscore = (2 * self.precision * self.recall)/(self.precision + self.recall)


class Evaluator:  # creates evaluation object+calculates macro/micro, no input
    def __init__(self, results=None, macro_fscore="", micro_fscore="", accuracy=""):
        if results is None:
            results = []
        self.results = results
        # list of objects from class Comparison() each responsible for different POS tags
        # e.g [NNresult_obj, VBresult_obj,...]
        self.macro_fscore = macro_fscore
        self.micro_fscore = micro_fscore
        self.accuracy = accuracy

    def evaluation(self, corpus_obj):
        # create objects of class Comparison(), compute P, R, F1 for each comparison_obj
        # and save all comparison_objects in self.results
        for domain_tag in corpus_obj.all_tags: # for every tag encountered in the data
            if domain_tag != "":
                domain_comparison = Comparison(domain_tag)  # makes compariscon_obj for pos_tag
                domain_comparison.get_comparison(corpus_obj)  # get {TP:x, FP:y,...}
                if domain_comparison.comparison['TP'] == 0: # if TP=0, then F1=0
                    domain_comparison.fscore = 0
                    domain_comparison.accuracy = 0
                else:
                    domain_comparison.get_precision()
                    domain_comparison.get_recall()
                    domain_comparison.get_fscore()
                    domain_comparison.get_accuracy()
                self.results.append(domain_comparison) # append to list of comparison_objects
        self.get_macro_fscore()
        self.get_micro_fscore()
        # compute macro and micro fscore and save inside a variable


    def get_macro_fscore(self):  # average fscores
        add_fscores = 0
        for result in self.results:
            add_fscores += result.fscore
        self.macro_fscore = add_fscores/len(self.results)

    def get_micro_fscore(self): # average TPs, FPs, FNs and TNs and compute P, R, F1
        add_results = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for result in self.results:  # adds all TPs... together
            add_results['TP'] += result.comparison['TP']
            add_results['FP'] += result.comparison['FP']
            add_results['FN'] += result.comparison['FN']
            add_results['TN'] += result.comparison['TN']
            total = result.comparison['TP']+result.comparison["TN"]+result.comparison['FN']+result.comparison["FP"]

        if add_results['TP'] == 0: # if TP=0, then F1=0
            self.micro_fscore = 0
        else:
            precision = add_results['TP']/(add_results['TP'] + add_results['FP'])
            recall = add_results['TP']/(add_results['TP'] + add_results['FN'])
            self.accuracy = add_results['TP']/total
            self.micro_fscore = (2 * precision * recall)/(precision + recall)


file = "clean_domain_data.json"
file = get_data(file)
corpus = Corpus(file)
corpus.create_objects()

# BASELINE 1: only 1 keyword
Hotel = JSONLookupDomain("Hotel")
Attraction = JSONLookupDomain("Attraction")
Restaurant = JSONLookupDomain("Restaurant")
Taxi = JSONLookupDomain("Taxi")
Train = JSONLookupDomain("Train")

dt = DomainTracker([Hotel, Attraction, Restaurant, Taxi, Train])

for dialogue in corpus.processed_corpus:
    dt.dialog_start()
    for sentence in dialogue.sentences:
        try:
            sentence.pred_domain = dt.select_domain(sentence.string)["predicted domain: "]
        except:
            sentence.pred_domain = []

evaluation = Evaluator()
evaluation.evaluation(corpus)
print("\n\nBASELINE 1: only one keyword")
print("micro_fscore: ", evaluation.micro_fscore, "macro_fscore: ", evaluation.macro_fscore, "accuracy: ", evaluation.accuracy)
for result in evaluation.results:
    print(result.domain, result.comparison, "fscore: ", result.fscore, "accuracy: ", result.accuracy)


# BASELINE 2: multiple hand-picked keywords

corpus2 = Corpus(file)
corpus2.create_objects()

Hotel2 = JSONLookupDomain("Hotel2")
Attraction2 = JSONLookupDomain("Attraction2")
Restaurant2 = JSONLookupDomain("Restaurant2")
Taxi2 = JSONLookupDomain("Taxi2")
Train2 = JSONLookupDomain("Train2")

dt = DomainTracker([Hotel2, Attraction2, Restaurant2, Taxi2, Train2])

for dialogue in corpus2.processed_corpus:
    dt.dialog_start()
    for sentence in dialogue.sentences:
        try:
            sentence.pred_domain = dt.select_domain(sentence.string)["predicted domain: "]
            sentence.pred_domain = [domain[:-1] for domain in sentence.pred_domain]
        except:
            sentence.pred_domain = []

evaluation2 = Evaluator()
evaluation2.evaluation(corpus2)
print("\n\nBASELINE 2: multiple hand-picked keywords")
print("micro_fscore: ", evaluation2.micro_fscore, "macro_fscore: ", evaluation2.macro_fscore, "accuracy: ", evaluation2.accuracy)
for result in evaluation2.results:
    print(result.domain, result.comparison, result.fscore)



""" to test our evaluator
data = {"x":[["h", [1]], ["i", [2]], ["j", [3]]]}
corpus = Corpus(data)
gold = []
for dialogue in corpus.processed_corpus:
    for i in range(len(gold)):
        dialogue[i].pred_domain = [gold[i]]"""