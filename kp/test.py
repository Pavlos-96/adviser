import ast

def get_data(file):
    f = open(file, "r")
    contents = f.read()
    dictionary = ast.literal_eval(contents)
    f.close()
    return dictionary

class Sentence:
    def __init__(self, string, pred_domain="", gold_domain=""):
        self.string = string
        self.pred_domain = pred_domain
        self.gold_domain = gold_domain


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
                sentence_obj.gold_domain = evaluation[1][0]
                self.all_tags.update({sentence_obj.gold_domain})
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

    def get_comparison(self, corpus):
        # for each domain find out the TPs FPs and FNs and save them in the self.comparison dictionary
        for dialogue in corpus.processed_corpus:
            for sentence in dialogue.sentences:
                print(sentence.gold_domain, sentence.pred_domain)
                if sentence.gold_domain == self.domain and sentence.pred_domain == self.domain:  # TP
                    self.comparison['TP'] += 1
                elif sentence.gold_domain != self.domain and sentence.pred_domain == self.domain:  # FP
                    self.comparison['FP'] += 1
                elif sentence.gold_domain == self.domain and sentence.pred_domain != self.domain:  # FN
                    self.comparison['FN'] += 1
                elif sentence.gold_domain != self.domain and sentence.pred_domain != self.domain:  #TN
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
    def __init__(self, results=None, macro_fscore="", micro_fscore=""):
        if results is None:
            results = []
        self.results = results
        # list of objects from class Comparison() each responsible for different POS tags
        # e.g [NNresult_obj, VBresult_obj,...]
        self.macro_fscore = macro_fscore
        self.micro_fscore = micro_fscore

    def evaluation(self, corpus_obj):
        # create objects of class Comparison(), compute P, R, F1 for each comparison_obj
        # and save all comparison_objects in self.results
        for domain_tag in corpus_obj.all_tags: # for every tag encountered in the data
            if domain_tag != "":
                domain_comparison = Comparison(domain_tag)  # makes compariscon_obj for pos_tag
                domain_comparison.get_comparison(corpus_obj)  # get {TP:x, FP:y,...}
                if domain_comparison.comparison['TP'] == 0: # if TP=0, then F1=0
                    domain_comparison.fscore = 0
                else:
                    domain_comparison.get_precision()
                    domain_comparison.get_recall()
                    domain_comparison.get_fscore()
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

        if add_results['TP'] == 0: # if TP=0, then F1=0
            self.micro_fscore = 0
        else:
            precision = add_results['TP']/(add_results['TP'] + add_results['FP'])
            recall = add_results['TP']/(add_results['TP'] + add_results['FN'])
            self.micro_fscore = (2 * precision * recall)/(precision + recall)


file = "clean_domain_data.json"
get_data(file)
test = {
    "PMUL1635.json": [
        [
            "I need to book a hotel in the east that has 4 stars.  ",
            [
                "Hotel"
            ]
        ],
        [
            "That doesn't matter as long as it has free wifi and parking.",
            [
                "Hotel"
            ]
        ],
        [
            "Could you book the Wartworth for one night, 1 person?",
            [
                "Hotel"
            ]
        ],
        [
            "Friday and Can you book it for me and get a reference number ?",
            [
                "Hotel"
            ]
        ],
        [
            "I am looking to book a train that is leaving from Cambridge to Bishops Stortford on Friday. ",
            [
                "Train"
            ]
        ],
        [
            "I want to get there by 19:45 at the latest. ",
            [
                "Train"
            ]
        ],
        [
            "Yes please. I also need the travel time, departure time, and price.",
            [
                "Train"
            ]
        ],
        [
            "Yes. Sorry, but suddenly my plans changed. Can you change the Wartworth booking to Monday for 3 people and 4 nights?",
            [
                "Hotel"
            ]
        ],
        [
            "Thank you very much, goodbye.",
            [
                "Hotel"
            ]
        ]
    ],
    "MUL2168.json": [
        [
            "Hi, I'm looking for a train that is going to cambridge and arriving there by 20:45, is there anything like that?",
            [
                "Train"
            ]
        ],
        [
            "I am departing from birmingham new street.",
            [
                "Train"
            ]
        ],
        [
            "I would like to leave on wednesday",
            [
                "Train"
            ]
        ],
        [
            "That will, yes. Please make a booking for 5 people please.",
            [
                "Train"
            ]
        ],
        [
            "Thanks so much. I would also need a place to say. I am looking for something with 4 stars and has free wifi. ",
            [
                "Hotel"
            ]
        ],
        [
            "That sounds great, could you make a booking for me please?",
            [
                "Hotel"
            ]
        ],
        [
            "Please book it for Wednesday for 5 people and 5 nights, please.",
            [
                "Hotel"
            ]
        ],
        [
            "Thank you, goodbye",
            [
                "Hotel"
            ]
        ]
    ]
}

corpus = Corpus(test)
corpus.create_objects()
for dialogue in corpus.processed_corpus:
    for sentence in dialogue.sentences:
        sentence.pred_domain = "Hotel"
evaluation = Evaluator()
evaluation.evaluation(corpus)
print("micro_fscore: ", evaluation.micro_fscore, "macro_fscore: ", evaluation.macro_fscore)
print(corpus.all_tags)