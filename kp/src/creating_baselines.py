import ast
import sys
import nltk
sys.path.insert(0, "/Users/pavlosmusenidis/Desktop/Computerlinguistik/2.Semester/SpokenDialogueSystems/adviser/adviser")
from Multiclass_Perceptron_domain_tracker import *


# BASELINE REQUIREMENTS
def setup_domaintracker():
    """Sets up a domain tracker with one keyword and returns it"""

    Hotel = JSONLookupDomain("Hotel")
    Attraction = JSONLookupDomain("Attraction")
    Restaurant = JSONLookupDomain("Restaurant")
    Taxi = JSONLookupDomain("Taxi")
    Train = JSONLookupDomain("Train")
    dt = DomainTracker([Hotel, Attraction, Restaurant, Taxi, Train])
    return dt

def setup_domaintracker_with_multiple_keywords():
    """Sets up a domain tracker with multiple keywords and returns it"""

    Hotel2 = JSONLookupDomain("Hotel2")
    Attraction2 = JSONLookupDomain("Attraction2")
    Restaurant2 = JSONLookupDomain("Restaurant2")
    Taxi2 = JSONLookupDomain("Taxi2")
    Train2 = JSONLookupDomain("Train2")
    dt = DomainTracker([Hotel2, Attraction2, Restaurant2, Taxi2, Train2])
    return dt

# BASELINE TAGGER
def tag(corpus, dt, multiple_keywords=0, wordnet=0, multiple_domains=0):
    """Tags a corpus depending on baseline setup
    Input:
        cropus: Object of class Corpus
        dt: Object of class DomainTracker
        multiple_keywords: Boolean, which shows if there are multiple keywords
        wordnet: Boolean, which shows if wordnet is used
        multiple_domains: Boolean, which shows if multiple domains are assigned"""

    for dialogue in corpus.processed_corpus:
        dt.dialog_start()
        for sentence in dialogue.sentences:
            if wordnet == 0 and multiple_domains == 0:
                domain = dt.select_domain_without_wordnet(sentence.string)
                if "predicted domain: " in domain and domain["predicted domain: "] != []:
                    if multiple_keywords == 0:
                        sentence.pred_domain = [domain["predicted domain: "][0]]
                    if multiple_keywords == 1:
                        sentence.pred_domain = [[domain[:-1] for domain in domain["predicted domain: "]][0]]
            elif wordnet == 1 and multiple_domains == 0:
                domain = dt.select_domain_with_wordnet(sentence.string)
                if "predicted domain: " in domain and domain["predicted domain: "] != []:
                    if multiple_keywords == 0:
                        sentence.pred_domain = [domain["predicted domain: "][0]]
                    if multiple_keywords == 1:
                        sentence.pred_domain = [[domain[:-1] for domain in domain["predicted domain: "]][0]]
            elif wordnet == 0 and multiple_domains == 1:
                domain = dt.select_domain_without_wordnet(sentence.string)
                if "predicted domain: " in domain:
                    if multiple_keywords == 0:
                        sentence.pred_domain = domain["predicted domain: "]
                    if multiple_keywords == 1:
                        sentence.pred_domain = [domain[:-1] for domain in domain["predicted domain: "]]
            elif wordnet == 1 and multiple_domains == 1:
                domain = dt.select_domain_with_wordnet(sentence.string)
                if "predicted domain: " in domain:
                    if multiple_keywords == 0:
                        sentence.pred_domain = domain["predicted domain: "]
                    if multiple_keywords == 1:
                        sentence.pred_domain = [domain[:-1] for domain in domain["predicted domain: "]]

def setup():
    """Sets up two corpora where one searches for one keyword for each domain
    and the other searches for multiple keywords"""

    # corpus with only one keyword
    corpus1 = Corpus(test)
    corpus1.create_objects()

    # corpus with multiple keywords
    corpus2 = Corpus(test)
    corpus2.create_objects()

    # with one keyword
    dt1 = setup_domaintracker()

    # with multiple keywords
    dt2 = setup_domaintracker_with_multiple_keywords()

    return corpus1, corpus2, dt1, dt2

def evaluate(corpus1, corpus2, f, wordnet=0, multiple_domains=0):
    """Does the evaluation and gives the baseline its title
    Input:
        corpus1: corpus with one keyword for each domain
        corpus2: corpus with multiple keywords for each domain
        f: file to save the results
        wordnet: Boolean, which shows if wordnet is used
        multiple_domains: Boolean, which shows if multiple domains are assigned"""

    evaluation1 = Evaluator()
    evaluation1.evaluation(corpus1)
    evaluation2 = Evaluator()
    evaluation2.evaluation(corpus2)
    if wordnet == 0 and multiple_domains == 0:
        title1 = "BASELINE 1: one keyword"
        title2 = "BASELINE 2: multiple keywords"
    elif wordnet == 0 and multiple_domains == 1:
        title1 = "BASELINE 3: one keyword, multiple assignments"
        title2 = "BASELINE 4: multiple keywords, multiple assignments"
    elif wordnet == 1 and multiple_domains == 0:
        title1 = "BASELINE 5: one keyword + wordnet synsets"
        title2 = "BASELINE 6: multiple keywords + wordnet synsets"
    elif wordnet == 1 and multiple_domains == 1:
        title1 = "BASELINE 7: one keyword + wordnet synsets, multiple assignments"
        title2 = "BASELINE 8: multiple keywords + wordnet synsets, multiple assignments"

    print(title1, "\nmacro-averaged fscore: ", evaluation1.macro_fscore,
          "\nmicro-averaged fscore", evaluation1.micro_fscore, "\naccuracy: ", evaluation1.accuracy, "\n")
    print(title1, "\nmacro-averaged fscore: ", evaluation1.macro_fscore,
          "\nmicro-averaged fscore", evaluation1.micro_fscore, "\naccuracy: ", evaluation1.accuracy, "\n", file=f)
    print(title2, "\nmacro-averaged fscore: ", evaluation1.macro_fscore,
          "\nmicro-averaged fscore", evaluation2.micro_fscore, "\naccuracy: ", evaluation2.accuracy, "\n")
    print(title2, "\nmacro-averaged fscore: ", evaluation2.macro_fscore,
          "\nmicro-averaged fscore", evaluation2.micro_fscore, "\naccuracy: ", evaluation2.accuracy, "\n", file=f)


if __name__ == "__main__":
    train = get_data("train.txt")
    test = get_data("test.txt")

    # CREATE ALL BASELINES
    f = open(str(Path(OUTPUT_DIRECTORY, "baselines.txt")), "w")
    for i in range(2):
        for j in range(2):
            corpus1, corpus2, dt1, dt2 = setup()
            tag(corpus1, dt1, 0, i, j)
            tag(corpus2, dt2, 1, i, j)
            evaluate(corpus1, corpus2, f, i, j)
    f.close()