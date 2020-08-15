import ast
import sys
import nltk
sys.path.insert(0, "/Users/pavlosmusenidis/Desktop/Computerlinguistik/2.Semester/SpokenDialogueSystems/adviser/adviser")
from Multiclass_Perceptron_domain_tracker import *


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
                        #print(sentence.pred_domain, sentence.gold_domain)
                    if multiple_keywords == 1:
                        sentence.pred_domain = [[domain[:-1] for domain in domain["predicted domain: "]][0]]
                        #print(sentence.pred_domain, sentence.gold_domain)
            elif wordnet == 1 and multiple_domains == 0:
                domain = dt.select_domain_with_wordnet(sentence.string)
                if "predicted domain: " in domain and domain["predicted domain: "] != []:
                    if multiple_keywords == 0:
                        sentence.pred_domain = [domain["predicted domain: "][0]]
                        #print(sentence.pred_domain, sentence.gold_domain)
                    if multiple_keywords == 1:
                        sentence.pred_domain = [[domain[:-1] for domain in domain["predicted domain: "]][0]]
                        #print(sentence.pred_domain, sentence.gold_domain)
            elif wordnet == 0 and multiple_domains == 1:
                domain = dt.select_domain_without_wordnet(sentence.string)
                if "predicted domain: " in domain:
                    if multiple_keywords == 0:
                        sentence.pred_domain = domain["predicted domain: "]
                        #print(sentence.pred_domain, sentence.gold_domain)
                    if multiple_keywords == 1:
                        sentence.pred_domain = [domain[:-1] for domain in domain["predicted domain: "]]
                        #print(sentence.pred_domain, sentence.gold_domain)
            elif wordnet == 1 and multiple_domains == 1:
                domain = dt.select_domain_with_wordnet(sentence.string)
                if "predicted domain: " in domain:
                    if multiple_keywords == 0:
                        sentence.pred_domain = domain["predicted domain: "]
                        #print(sentence.pred_domain, sentence.gold_domain)
                    if multiple_keywords == 1:
                        sentence.pred_domain = [domain[:-1] for domain in domain["predicted domain: "]]
                        #print(sentence.pred_domain, sentence.gold_domain)


if __name__ == "__main__":
    train = get_data("train.txt")
    test = get_data("test.txt")
    # BASELINES

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

    tag(corpus1, dt1, 0, 1, 1)
    tag(corpus2, dt2, 1, 1, 1)

    evaluation1 = Evaluator()
    evaluation1.evaluation(corpus1)
    print("with one keyword\nmacro-averaged fscore: ", evaluation1.macro_fscore,
          "\nmicro-averaged fscore", evaluation1.micro_fscore, "\naccuracy: ", evaluation1.accuracy)

    evaluation2 = Evaluator()
    evaluation2.evaluation(corpus2)
    print("with multiple keywords\nmacro-averaged fscore: ", evaluation2.macro_fscore,
          "\nmicro-averaged fscore", evaluation2.micro_fscore, "\naccuracy: ", evaluation2.accuracy)