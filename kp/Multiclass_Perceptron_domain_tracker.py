import ast
import sys
import nltk
sys.path.insert(0, "/Users/pavlosmusenidis/Desktop/Computerlinguistik/2.Semester/SpokenDialogueSystems/adviser/adviser")

from services.service import Service, PublishSubscribe, DialogSystem
from domain_tracker import DomainTracker
from utils.domain import Domain
from utils.domain.jsonlookupdomain import JSONLookupDomain
from sklearn.metrics import f1_score
import random

def get_data(file):
    f = open(file, "r")
    contents = f.read()
    dictionary = ast.literal_eval(contents)
    f.close()
    return dictionary

class Sentence:
    def __init__(self, string, pred_domain="", gold_domain="", features=None,
                 highest_score=None):
        self.string = string # "I want wifi"
        self.pred_domain = pred_domain # "Train"
        self.gold_domain = gold_domain # "Hotel"
        if features is None:
            features = []
        self.features = features
        self.highest_score = highest_score


class Dialogue:
    def __init__(self, sentences=None):
        if sentences is None:
            sentences = list()
        self.sentences = sentences

    def features(self, all_tags, keywords): # we get the features on sentence level, so we can use features like W-1 and W+1
        for i in range(len(self.sentences)):
            sentence = nltk.word_tokenize(self.sentences[i].string.lower())
            try:
                prev_sentence = nltk.word_tokenize(self.sentences[i-1].string.lower())
            except:
                pass
            try:
                next_sentence = nltk.word_tokenize(self.sentences[i+1].string.lower())
            except:
                pass
            try:
                pre_prev_sentence = nltk.word_tokenize(self.sentences[i - 1].string.lower())
            except:
                pass
            try:
                next_next_sentence = nltk.word_tokenize(self.sentences[i+1].string.lower())
            except:
                pass

            # find features:

            # words
            for word in sentence:
                if f"{word} in sentence" not in self.sentences[i].features:
                    self.sentences[i].features.append(f"{word.lower()} in sentence")

            # question
            if "?" in sentence:
                self.sentences[i].features.append("? in sentence")

            # label in sentence
            for label in all_tags:
                if label.lower() in sentence:
                    if f"{label.lower()}-label in sentence" not in self.sentences[i].features:
                        self.sentences[i].features.append(f"{label.lower()}-label in sentence")

            # label in sentence-1
            try:
                for label in all_tags:
                    if label.lower() in prev_sentence:
                        self.sentences[i].features.append(f"{label.lower()}-label in sentence-1")
            except:
                self.sentences[i].features.append("dialogue-start")
                pass

            # label in sentence+1
            try:
                for label in all_tags:
                    if label.lower() in next_sentence:
                        self.sentences[i].features.append(f"{label.lower()}-label in sentence+1")
            except:
                self.sentences[i].features.append("dialogue-end")
                pass

            # label in sentence-2
            try:
                for label in all_tags:
                    if label.lower() in pre_prev_sentence:
                        self.sentences[i].features.append(f"{label.lower()}-label in sentence-2")
            except:
                pass

            # label in sentence+2
            try:
                for label in all_tags:
                    if label.lower() in next_next_sentence:
                        self.sentences[i].features.append(f"{label.lower()}-label in sentence+2")
            except:
                pass

            # keyword in sentence
            for keyword in keywords:
                if keyword.lower() in sentence:
                    if f"{keyword.lower()}-keyword in sentence" not in self.sentences[i].features:
                        self.sentences[i].features.append(f"{keyword.lower()}-keyword in sentence")

            # keyword in sentence-1
            try:
                for keyword in keywords:
                    if keyword.lower() in prev_sentence:
                        self.sentences[i].features.append(f"{keyword.lower()}-keyword in sentence-1")
            except:
                self.sentences[i].features.append("dialogue-start")
                pass

            # keyword in sentence+1
            try:
                for keyword in keywords:
                    if keyword.lower() in next_sentence:
                        self.sentences[i].features.append(f"{keyword.lower()}-keyword in sentence+1")
            except:
                self.sentences[i].features.append("dialogue-end")
                pass

            # keyword in sentence-2
            try:
                for keyword in keywords:
                    if keyword.lower() in pre_prev_sentence:
                        self.sentences[i].features.append(f"{keyword.lower()}-keyword in sentence-2")
            except:
                pass

            # keyword in sentence+2
            try:
                for keyword in keywords:
                    if keyword.lower() in next_next_sentence:
                        self.sentences[i].features.append(f"{keyword.lower()}-keyword in sentence+2")
            except:
                pass

            # special feature
            self.sentences[i].features.append("BIAS")


class Corpus:  # input is dictionary with data
    def __init__(self, dictionary, all_tags=None, processed_corpus=None,
                 all_features=None, keywords = None):
        self.dictionary = dictionary
        if all_tags is None:
            all_tags = set()
        self.all_tags = all_tags  # set of all tags in gold + predicted
        if processed_corpus is None:
            processed_corpus = list()
        self.processed_corpus = processed_corpus
        if all_features is None:
            all_features = set()
        self.all_features = all_features  # set of feature types
        if keywords is None:
            keywords = ["trip", "museum", "town", "visit", "city", "club", "see", "entertain",
            "stay", "guest", "night", "wifi", "free", "parking", "room", "sleep",
            "table", "grill", "mediterranean", "oriental", "asian", "chinese", "food", "indian", "eat", "dine"
            "pick", "car", "cab",
            "leaving", "depart", "arrive", "travel", "schedule",
            "want", "need", "from", "to", "get", "leave", "book", "destination"]
        self.keywords = keywords

    def create_objects(self):  # make sentence + sentence objects, no input
        for data in self.dictionary:  # e.g ['We', 'are', 'happy', '.']
            sentences = []  # list of sentence objects
            for evaluation in self.dictionary[data]:  # sentence
                sentence_obj = Sentence(evaluation[0])  # creates sentence
                sentence_obj.gold_domain = evaluation[1]
                if len(sentence_obj.gold_domain) != 0:
                    for tag in sentence_obj.gold_domain:
                        self.all_tags.update({tag})
                sentences.append(sentence_obj)
            dialogue_obj = Dialogue(sentences)  # creates sentence object
            dialogue_obj.features(self.all_tags, self.keywords)
            self.processed_corpus.append(dialogue_obj)
        self.get_features()  # collects all features

    def get_features(self):  # retrieve all features from the sentences
        for dialogue in self.processed_corpus:
            for sentence in dialogue.sentences:
                self.all_features.update(set(sentence.features))

    def relabel(self):
        for dialogue in self.processed_corpus:
            for i in range(len(dialogue.sentences)):
                if dialogue.sentences[i].pred_domain == ["no transition"]:
                    try:
                        dialogue.sentences[i].pred_domain = dialogue.sentences[i-1].pred_domain
                    except:
                        dialogue.sentences[i].pred_domain = []

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

    def evaluation(self, corpus_obj):
        # create objects of class Comparison(), compute P, R, F1 for each comparison_obj
        # and save all comparison_objects in self.results
        for domain_tag in corpus_obj.all_tags:  # for every tag encountered in the data
            if domain_tag != "":
                domain_comparison = Comparison(domain_tag)  # makes compariscon_obj for pos_tag
                domain_comparison.get_comparison(corpus_obj)  # get {TP:x, FP:y,...}
                if domain_comparison.comparison['TP'] == 0:  # if TP=0, then F1=0
                    domain_comparison.fscore = 0
                    domain_comparison.accuracy = 0
                else:
                    domain_comparison.get_precision()
                    domain_comparison.get_recall()
                    domain_comparison.get_fscore()
                    domain_comparison.get_accuracy()
                self.results.append(domain_comparison)  # append to list of comparison_objects
        self.get_macro_fscore()
        self.get_micro_fscore()
        # compute macro and micro fscore and save inside a variable

# BASELINE REQUIREMENTS
def setup_domaintracker():
    Hotel = JSONLookupDomain("Hotel")
    Attraction = JSONLookupDomain("Attraction")
    Restaurant = JSONLookupDomain("Restaurant")
    Taxi = JSONLookupDomain("Taxi")
    Train = JSONLookupDomain("Train")
    dt = DomainTracker([Hotel, Attraction, Restaurant, Taxi, Train])
    return dt

def setup_domaintracker_with_multiple_keywords():
    Hotel2 = JSONLookupDomain("Hotel2")
    Attraction2 = JSONLookupDomain("Attraction2")
    Restaurant2 = JSONLookupDomain("Restaurant2")
    Taxi2 = JSONLookupDomain("Taxi2")
    Train2 = JSONLookupDomain("Train2")
    dt = DomainTracker([Hotel2, Attraction2, Restaurant2, Taxi2, Train2])
    return dt

# BASELINE TAGGER
def tag(corpus, dt, multiple_keywords=0, wordnet=0, multiple_domains=0):
    for dialogue in corpus.processed_corpus:
        dt.dialog_start()
        for sentence in dialogue.sentences:
            if wordnet == 0 and multiple_domains == 0:
                domain = dt.select_domain_without_wordnet(sentence.string)
                if "predicted domain: " in domain and domain["predicted domain: "] != []:
                    if multiple_keywords == 0:
                        sentence.pred_domain = [domain["predicted domain: "][0]]
                        print(sentence.pred_domain, sentence.gold_domain)
                    if multiple_keywords == 1:
                        sentence.pred_domain = [[domain[:-1] for domain in domain["predicted domain: "]][0]]
                        print(sentence.pred_domain, sentence.gold_domain)
            elif wordnet == 1 and multiple_domains == 0:
                domain = dt.select_domain_with_wordnet(sentence.string)
                if "predicted domain: " in domain and domain["predicted domain: "] != []:
                    if multiple_keywords == 0:
                        sentence.pred_domain = [domain["predicted domain: "][0]]
                        print(sentence.pred_domain, sentence.gold_domain)
                    if multiple_keywords == 1:
                        sentence.pred_domain = [[domain[:-1] for domain in domain["predicted domain: "]][0]]
                        print(sentence.pred_domain, sentence.gold_domain)
            elif wordnet == 0 and multiple_domains == 1:
                domain = dt.select_domain_without_wordnet(sentence.string)
                if "predicted domain: " in domain:
                    if multiple_keywords == 0:
                        sentence.pred_domain = domain["predicted domain: "]
                        print(sentence.pred_domain, sentence.gold_domain)
                    if multiple_keywords == 1:
                        sentence.pred_domain = [domain[:-1] for domain in domain["predicted domain: "]]
                        print(sentence.pred_domain, sentence.gold_domain)
            elif wordnet == 1 and multiple_domains == 1:
                domain = dt.select_domain_with_wordnet(sentence.string)
                if "predicted domain: " in domain:
                    if multiple_keywords == 0:
                        sentence.pred_domain = domain["predicted domain: "]
                        print(sentence.pred_domain, sentence.gold_domain)
                    if multiple_keywords == 1:
                        sentence.pred_domain = [domain[:-1] for domain in domain["predicted domain: "]]
                        print(sentence.pred_domain, sentence.gold_domain)


class Perceptron():  # input = pos tag e.g. NN
    def __init__(self, transition, weights=None):
        self.transition = transition
        if weights is None:
            weights = dict()
        self.weights = weights  # {feature1:0, feature2:0, ...}

    def create_weights(self, all_features): # for each feature initialize weight with random values between -1 and 1
        for feature in all_features:
        # all_features = corpus_obj.all_features
            self.weights[feature] = (random.random()*2-1)
            # e.g {feature1:0.64, feature2:-0.32,...}

    def assign_pred(self, sentence):  # method gets called for each sentence to assign sentence.pred_tag
        competing_score = 0
        for feature in sentence.features: # look at sentence_features
            if feature in self.weights: # look if weights exist for the sentence_features
                competing_score += self.weights[feature] # calculate score
        if sentence.highest_score is None or competing_score >= sentence.highest_score:
            # if score of this perceptron is higher
            sentence.highest_score = competing_score # save score in sentence
            sentence.pred_domain = [self.transition] # assign perceptrons' label tag to sentence


class Multiclass_perceptron():
    def __init__(self, iterations=30, perceptrons=None, evaluations=None):
        self.iterations = iterations
        if perceptrons is None:
            perceptrons = dict()
        self.perceptrons = perceptrons
        # dict of POS tags to perceptron_objects, e.g. {NN:perceptron1, VB:perceptron2,...}
        if evaluations is None:
            evaluations = dict()
        self.evaluations = evaluations
        # dict of iterations to evaluations, e.g. {0:evaluation1, 1:evaluation2,...}


    def start_perceptrons(self, corpus):  # create perceptron for each POS tag and initialize weights
        for tag in corpus.all_tags: # for every seen tag
            domain_perceptron = Perceptron(tag) # create perceptron_obj
            domain_perceptron.create_weights(corpus.all_features)
            # create dict from features to weights of 0, e.g. {feature1:0, feature2:0,...}
            self.perceptrons[tag] = domain_perceptron # append to dict from POS tags to perceptron_objects
            # e.g {NN:perceptron1, VB:perceptron2,...}
        domain_perceptron = Perceptron("no transition")  # create perceptron_obj
        domain_perceptron.create_weights(corpus.all_features)
        # create dict from features to weights of 0, e.g. {feature1:0, feature2:0,...}
        self.perceptrons["no transition"] = domain_perceptron  # append to dict from POS tags to perceptron_objects

    def predict(self, corpus, training=True): # predict most likely POS tag for each sentence
        if training is False:
            for dialogue in corpus.processed_corpus:
                for sentence in dialogue.sentences:
                    sentence.highest_score = None
        for tag in self.perceptrons:  # for each perceptron
            for dialogue in corpus.processed_corpus:
                for sentence in dialogue.sentences:  # for each sentence
                    self.perceptrons[tag].assign_pred(sentence)
                    # if score of this perceptron higher, assign its pred_tag to sentence
        if training is False:
            corpus.relabel()
            evaluation = Evaluator()
            evaluation.evaluation(corpus)
            print("prediction:   macro:", evaluation.macro_fscore, "micro:", evaluation.micro_fscore, "accuracy:", evaluation.accuracy)
            for result in evaluation.results:
                print(result.domain, result.comparison, "fscore: ", result.fscore)

    def adjust_weights(self, corpus, iteration):
        for dialogue in corpus.processed_corpus:
            for i in range(len(dialogue.sentences)): # for each sentence
                dialogue.sentences[i].highest_score = None
                if dialogue.sentences[i].gold_domain != dialogue.sentences[i-1].gold_domain:
                    if "no transition" in dialogue.sentences[i].pred_domain:
                        for feature in dialogue.sentences[i].features:
                            self.perceptrons["no transition"].weights[feature] -= 1 - iteration/self.iterations
                            self.perceptrons[dialogue.sentences[i].gold_domain[0]].weights[feature] += 1 - iteration/self.iterations
                    else:
                        for domain in dialogue.sentences[i].pred_domain:
                            if domain not in dialogue.sentences[i].gold_domain:
                                for feature in dialogue.sentences[i].features:
                                    self.perceptrons[domain].weights[feature] -= 1 - iteration/self.iterations
                                    self.perceptrons[dialogue.sentences[i].gold_domain[0]].weights[feature] += 1 - iteration/self.iterations
                elif dialogue.sentences[i].gold_domain == dialogue.sentences[i-1].gold_domain:
                    if "no transition" not in dialogue.sentences[i].pred_domain:
                        for feature in dialogue.sentences[i].features:
                            self.perceptrons[dialogue.sentences[i].pred_domain[0]].weights[feature] -= 1 - iteration/self.iterations
                            self.perceptrons["no transition"].weights[feature] += 1 - iteration/self.iterations

    def train(self, corpus):
        self.start_perceptrons(corpus) # create perceptrons and their weights
        for i in range(self.iterations):
            self.predict(corpus) # predict label tag for each sentence
            if i != self.iterations - 1:
                self.adjust_weights(corpus, i) # adjust weights
            corpus.relabel()
            evaluation = Evaluator()
            evaluation.evaluation(corpus) # do evaluation
            print(i+1, ":  macro:", evaluation.macro_fscore, "micro:", evaluation.micro_fscore, "accuracy:", evaluation.accuracy)
            self.evaluations[i] = evaluation # save evaluations by iterations

    def test(self,corpus):
        self.predict(corpus, False)


if __name__ == "__main__":
    train = get_data("train.txt")
    test = get_data("test.txt")

    # TAGGER
    corpus_obj = Corpus(train)
    corpus_obj.create_objects()

    # train classifier
    mc_perceptron = Multiclass_perceptron()
    print("starting to train")
    mc_perceptron.train(corpus_obj)

    # create test corpus:
    test_corpus = Corpus(test)
    test_corpus.create_objects()

    # use classifier trained classifier on test_set
    mc_perceptron.test(test_corpus)



