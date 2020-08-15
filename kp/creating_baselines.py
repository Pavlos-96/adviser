import ast
import sys
import nltk
sys.path.insert(0, "/Users/pavlosmusenidis/Desktop/Computerlinguistik/2.Semester/SpokenDialogueSystems/adviser/adviser")
from Multiclass_Perceptron_domain_tracker import *


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