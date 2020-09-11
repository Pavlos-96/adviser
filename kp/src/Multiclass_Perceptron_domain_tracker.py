import ast
import sys
import nltk
sys.path.insert(0, "/Users/pavlosmusenidis/Desktop/Computerlinguistik/2.Semester/SpokenDialogueSystems/adviser/adviser")
from pathlib import Path


DATA_DIRECTORY = "data"
OUTPUT_DIRECTORY = "results"


import random

def get_data(file):
    """Gets data from a file and saves it as a dictionary"""

    f = open(str(Path(DATA_DIRECTORY, file)), "r")
    contents = f.read()
    dictionary = ast.literal_eval(contents)
    f.close()
    return dictionary

class Sentence:
    """Objects are sentences.
    Contains variables for making and saving predictions.
    Input:
        string: String of the sentence"""

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
    """The objects are sentences.
    Useful for generating features with context"""

    def __init__(self, sentences=None):
        if sentences is None:
            sentences = list()
        self.sentences = sentences

    def features(self, all_tags, keywords, generalizable=True):
        """Converts the preprocessed text, so that sentences are objects
        containing lists of token objects
        Input:
            all_tags: Set of all domains in the training data
            keywords: List of keywords
            generalizable: If True feature set 1 and if False feature set 2 is activated"""

        for i in range(len(self.sentences)):
            sentence = nltk.word_tokenize(self.sentences[i].string.lower())
            try:
                prev_sentence = nltk.word_tokenize(self.sentences[i-1].string.lower())
            except:
                pass
            try:
                pre_prev_sentence = nltk.word_tokenize(self.sentences[i-1].string.lower())
            except:
                pass

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
                        if f"{label.lower()}-label in sentence-1" not in self.sentences[i].features:
                            self.sentences[i].features.append(f"{label.lower()}-label in sentence-1")
            except:
                self.sentences[i].features.append("dialogue-start")
                pass

            # label in sentence-2
            try:
                for label in all_tags:
                    if label.lower() in pre_prev_sentence:
                        self.sentences[i].features.append(f"{label.lower()}-label in sentence-2")
            except:
                pass

            if generalizable == False:
                # keyword in sentence
                for keyword in keywords:
                    if keyword.lower() in sentence:
                        if f"{keyword.lower()}-keyword in sentence" not in self.sentences[i].features:
                            self.sentences[i].features.append(f"{keyword.lower()}-keyword in sentence")

                # keyword in sentence-1
                try:
                    for keyword in keywords:
                        if keyword.lower() in prev_sentence:
                            if f"{keyword.lower()}-keyword in sentence-1" not in self.sentences[i].features:
                                self.sentences[i].features.append(f"{keyword.lower()}-keyword in sentence-1")

                except:
                    self.sentences[i].features.append("dialogue-start")
                    pass

                # keyword in sentence-2
                try:
                    for keyword in keywords:
                        if keyword.lower() in pre_prev_sentence:
                            if f"{keyword.lower()}-keyword in sentence-2" not in self.sentences[i].features:
                                self.sentences[i].features.append(f"{keyword.lower()}-keyword in sentence-2")

                except:
                    pass

            # special feature
            self.sentences[i].features.append("BIAS")


class Corpus:
    """The objects are corpora.
    Includes variables with information on the corpus
    and methods to do operations on corpus level.
    Input:
        dictionary: A dictionary containing the data"""

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

    def create_objects(self, generalizable):
        """Converts the preprocessed text, so that sentences are objects
        containing lists of token objects
        Input:
            generalizable: If True feature set 2 is activated"""

        for data in self.dictionary:
            sentences = []
            for evaluation in self.dictionary[data]:
                sentence_obj = Sentence(evaluation[0])
                sentence_obj.gold_domain = evaluation[1]
                if len(sentence_obj.gold_domain) != 0:
                    for tag in sentence_obj.gold_domain:
                        self.all_tags.update({tag})
                sentences.append(sentence_obj)
            dialogue_obj = Dialogue(sentences)
            dialogue_obj.features(self.all_tags, self.keywords, generalizable)
            self.processed_corpus.append(dialogue_obj)
        self.get_features()  # collects all features

    def get_features(self):
        """Collects all features and saves them in a class variable"""

        for dialogue in self.processed_corpus:
            for sentence in dialogue.sentences:
                self.all_features.update(set(sentence.features))

    def relabel(self):
        """Relabels dialogues, which were temporarily labeled with 'no transition'."""

        for dialogue in self.processed_corpus:
            for i in range(len(dialogue.sentences)):
                if dialogue.sentences[i].pred_domain == ["no transition"]:
                    try:
                        dialogue.sentences[i].pred_domain = dialogue.sentences[i-1].pred_domain
                    except:
                        dialogue.sentences[i].pred_domain = []

class Comparison:
    """Objects of this class contain results of the evaluation of specific POS tags.
    Input:
        domain = domain label"""

    def __init__(self, domain,
                 comparison=None,
                 precision=0, recall=0, accuracy=0, fscore=0):
        self.domain = domain
        if comparison is None:
            comparison = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        self.comparison = comparison
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.fscore = fscore

    def get_comparison(self, corpus):
        """Finds out for each domain the TPs FPs and FNs and saves them in the self.comparison dictionary
        Input:
            cropus_obj: Object of class Corpus"""

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
        """Calculates Precision"""

        self.precision = self.comparison['TP']/(self.comparison['TP'] + self.comparison['FP'])

    def get_recall(self):
        """Calculates Recall"""

        self.recall = self.comparison['TP']/(self.comparison['TP'] + self.comparison['FN'])

    def get_accuracy(self):
        """Calculates Accuracy"""

        self.accuracy = (self.comparison['TP']+self.comparison['TN'])/\
                        (self.comparison['TP'] +self.comparison['TN'] + self.comparison['FP'] + self.comparison['FN'])

    def get_fscore(self):
        """Calculates F1-Score"""

        self.fscore = (2 * self.precision * self.recall)/(self.precision + self.recall)


class Evaluator:
    """Objects of this class contain the results of an evaluation and
    contain methods to calculate macro- and micro-averaged F1-Scores"""

    def __init__(self, results=None, macro_fscore="", micro_fscore="", accuracy=0):
        if results is None:
            results = []
        self.results = results
        self.macro_fscore = macro_fscore
        self.micro_fscore = micro_fscore
        self.accuracy = accuracy


    def get_macro_fscore(self):
        """Calculates macro-averaged F1-Score"""

        add_fscores = 0
        for result in self.results:
            add_fscores += result.fscore
        self.macro_fscore = add_fscores/len(self.results)

    def get_micro_fscore_and_accuracy(self): # average TPs, FPs, FNs and TNs and compute P, R, F1
        """Calculates micro-averaged F1-Score and accuracy"""

        add_results = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for result in self.results:
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
        """Does the evaluation and saves the results
        Input:
            corpus_obj: Object of class Corpus"""

        for domain_tag in corpus_obj.all_tags:
            if domain_tag != "":
                domain_comparison = Comparison(domain_tag)
                domain_comparison.get_comparison(corpus_obj)
                if domain_comparison.comparison['TP'] == 0:
                    domain_comparison.fscore = 0
                    domain_comparison.accuracy = 0
                else:
                    domain_comparison.get_precision()
                    domain_comparison.get_recall()
                    domain_comparison.get_fscore()
                    domain_comparison.get_accuracy()
                self.results.append(domain_comparison)
        self.get_macro_fscore()
        self.get_micro_fscore_and_accuracy()


class Perceptron():
    """Objects of this class include variables where weights
    for each feature are stored and methods to adjust them.
    Input:
        transition = transition tag"""

    def __init__(self, transition, weights=None):
        self.transition = transition
        if weights is None:
            weights = dict()
        self.weights = weights

    def create_weights(self, all_features):
        """Initializes random weights for all features.
        Input:
            all_features: List of all features"""

        for feature in all_features:
            self.weights[feature] = (random.random()*2-1)

    def assign_pred(self, sentence):
        """Calculates score for a sentence and assigns
        the transition label, if it is the highest score.
        Input:
            sentence: Object of class Sentence"""

        competing_score = 0
        for feature in sentence.features: # look at sentence_features
            if feature in self.weights: # look if weights exist for the sentence_features
                competing_score += self.weights[feature] # calculate score
        if sentence.highest_score is None or competing_score >= sentence.highest_score:
            # if score of this perceptron is higher
            sentence.highest_score = competing_score # save score in sentence
            sentence.pred_domain = [self.transition] # assign perceptrons' transition label to sentence


class Multiclass_perceptron():
    """Object is the Multiclass Perceptron.
    It contains variables storing perceptrons and evaluations
    Input:
        tags: Set of all domains in the training data
        keywords: List of keywords
        generalizable: If True feature set 1 and if False feature set 2 is activated"""

    def __init__(self, tags, keywords, generalizable, iterations=40, context=None, perceptrons=None, evaluations=None):
        self.tags = tags
        self.keywords = keywords
        self.generelizable = generalizable
        self.iterations = iterations
        if tags is None:
            tags = set()
        self.tags = tags
        if context is None:
            context = Dialogue([])
        self.context = context
        if perceptrons is None:
            perceptrons = dict()
        self.perceptrons = perceptrons
        if evaluations is None:
            evaluations = dict()
        self.evaluations = evaluations


    def start_perceptrons(self, corpus):
        """Creates perceptron for each possible domain transition and for cases of no
        transition and initializes weights
        Input:
        corpus: Object of class Corpus"""

        for tag in corpus.all_tags:
            domain_perceptron = Perceptron(tag)
            domain_perceptron.create_weights(corpus.all_features)
            self.perceptrons[tag] = domain_perceptron
        domain_perceptron = Perceptron("no transition")
        domain_perceptron.create_weights(corpus.all_features)
        self.perceptrons["no transition"] = domain_perceptron

    def predict(self, corpus, f, training=True):
        """Assigns a transition label to each sentence in the corpus, by taking
        the transition label of the perceptron with the highest score.
        Input:
            corpus: Object of class Corpus
            f: file to save results
            training: Boolean, to differentiates between training and testing"""

        if training is False:
            for dialogue in corpus.processed_corpus:
                for sentence in dialogue.sentences:
                    sentence.highest_score = None
        for tag in self.perceptrons:
            for dialogue in corpus.processed_corpus:
                for sentence in dialogue.sentences:
                    self.perceptrons[tag].assign_pred(sentence)
                    # if score of this perceptron higher, assign its pred_tag to sentence
        if training is False:
            corpus.relabel()
            evaluation = Evaluator()
            evaluation.evaluation(corpus)
            print("prediction:\tmacro:", evaluation.macro_fscore, "\tmicro:", evaluation.micro_fscore,
                  "\taccuracy:", evaluation.accuracy)
            print("prediction:\tmacro:", evaluation.macro_fscore, "\tmicro:", evaluation.micro_fscore,
                  "\taccuracy:", evaluation.accuracy, file=f)
            for result in evaluation.results:
                print(result.domain, result.comparison, "fscore: ", result.fscore)

    def domain_guesser(self):
        """This activates the interactive version of the domain tracker."""

        print("Talk to me about a topic \
        and i will guess which topic you are talking about!")
        print("type 'new' to start a new dialog or 'exit' to exit the domain guesser")
        while True:
            print(self.tags)
            utterance = input()
            if utterance == "new":
                self.context = Dialogue()
            elif utterance == "exit":
                break
            else:
                self.predict_in_real_time(self, utterance)

    def predict_in_real_time(self, utterance):
        """Can be used for real-time predictions.
        This could be the main function of the trained domain tracker
        Input:
            utterance: The utterance the domain shall be predicted for"""

        if utterance == "new":
            self.context = Dialogue()
        sentence = Sentence(utterance)
        self.context.sentences.append(sentence)
        self.context.features(self.tags, self.keywords, self.generelizable)
        utterance = self.context.sentences[-1]
        for tag in self.tags:
            self.perceptrons[tag].assign_pred(self.context.sentences[-1])
        if "no transition" in utterance.pred_domain:
            if len(self.context.sentences) >= 2:
                utterance.pred_domain = [self.context.sentences[-2].pred_domain[0]]
                domain = utterance.pred_domain[0]
                print(domain)
            else:
                print("not found")
        else:
            domain = utterance.pred_domain[0]
            print(domain)


    def adjust_weights(self, corpus, iteration):
        """Adjust weights, depending on the correctness of the prediction.
        Input:
            corpus: Object of class Corpus"""

        for dialogue in corpus.processed_corpus:
            for i in range(len(dialogue.sentences)): # for each sentence
                dialogue.sentences[i].highest_score = None
                if dialogue.sentences[i].gold_domain != dialogue.sentences[i-1].gold_domain:
                    if "no transition" in dialogue.sentences[i].pred_domain:
                        for feature in dialogue.sentences[i].features:
                            self.perceptrons["no transition"].weights[feature] \
                                -= 1 - iteration/self.iterations
                            self.perceptrons[dialogue.sentences[i].gold_domain[0]].weights[feature] \
                                += 1 - iteration/self.iterations
                    else:
                        for domain in dialogue.sentences[i].pred_domain:
                            if domain not in dialogue.sentences[i].gold_domain:
                                for feature in dialogue.sentences[i].features:
                                    self.perceptrons[domain].weights[feature] \
                                        -= 1 - iteration/self.iterations
                                    self.perceptrons[dialogue.sentences[i].gold_domain[0]].weights[feature] \
                                        += 1 - iteration/self.iterations
                elif dialogue.sentences[i].gold_domain == dialogue.sentences[i-1].gold_domain:
                    if "no transition" not in dialogue.sentences[i].pred_domain:
                        for feature in dialogue.sentences[i].features:
                            self.perceptrons[dialogue.sentences[i].pred_domain[0]].weights[feature] \
                                -= 1 - iteration/self.iterations
                            self.perceptrons["no transition"].weights[feature] \
                                += 1 - iteration/self.iterations

    def train(self, corpus, f):
        """Trains the Multiclass Perceptron.
        Input:
            corpus: Object of class Corpus
            f: file to save results"""

        self.start_perceptrons(corpus) # create perceptrons and their weights
        for i in range(self.iterations):
            self.predict(corpus, f) # predict label tag for each sentence
            if i != self.iterations - 1:
                self.adjust_weights(corpus, i) # adjust weights
            corpus.relabel()
            evaluation = Evaluator()
            evaluation.evaluation(corpus) # do evaluation
            print(f"{i+1}:\tmacro:", evaluation.macro_fscore, "\tmicro:", evaluation.micro_fscore,
                  "\taccuracy:", evaluation.accuracy)
            print(f"{i+1}:\tmacro:", evaluation.macro_fscore, "\tmicro:", evaluation.micro_fscore,
                  "\taccuracy:", evaluation.accuracy, file=f)
            self.evaluations[i] = evaluation # save evaluations by iterations

    def test(self,corpus, f):
        """Does the final prediction on the test corpus
        Input:
        corpus: Object of class Corpus
        f: file to save results"""

        self.predict(corpus, f, False)


if __name__ == "__main__":
    train = get_data("train.txt")
    test = get_data("test.txt")
    if sys.argv[1] == "FS2":
        generalizable = False
        f = open(str(Path(OUTPUT_DIRECTORY, "Multiclass_Perceptron_results_FS2.txt")), "w")
    else:
        generalizable = True
        f = open(str(Path(OUTPUT_DIRECTORY, "Multiclass_Perceptron_results_FS1.txt")), "w")

    # TAGGER
    corpus_obj = Corpus(train)
    corpus_obj.create_objects(generalizable)

    # train classifier
    mc_perceptron = Multiclass_perceptron(corpus_obj.all_tags, corpus_obj.keywords, generalizable)
    print("starting to train")
    mc_perceptron.train(corpus_obj, f)

    # create test corpus:
    test_corpus = Corpus(test)
    test_corpus.create_objects(generalizable)

    # use classifier trained classifier on test_set
    mc_perceptron.test(test_corpus, f)
    f.close()



