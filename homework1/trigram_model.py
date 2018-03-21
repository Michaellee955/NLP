import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Spring 2018
Homework 1 - Programming Component: Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    length = len(sequence)
    #if only require 1-gram, then we need to add one START and one END to the sequence. 
    if n==1 or n==2:
        sequence=["START"]*n+sequence+["STOP"]
        end = n+1  #end i means that when n==1, we need to read one more data, that is to the end of sequence, which is slightly different from when n>1.
    #if require multi-grams, use the common calculation below.
    else:
        sequence = ["START"]*(n-1)+sequence+["STOP"]
        end = 1
    if n==2:
        end = n
    result = []
    temp = ()
    #the process to construct the tuple-based array.
    for i in range(0,length+end):
        temp = tuple(sequence[i:i+n])

        result.append(temp)
    return result


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        self.total = 2
        ##Your code here

        for sentence in corpus:
            temp_1 = get_ngrams(sentence,1)
            temp_2 = get_ngrams(sentence,2)
            temp_3 = get_ngrams(sentence,3)
            for i in range(len(temp_1)):
                if temp_1[i] in self.unigramcounts:
                    self.unigramcounts[temp_1[i]] += 1
                else:
                    self.unigramcounts[temp_1[i]] = 1
                self.total += 1

            for i in range(len(temp_2)):
                if temp_2[i] in self.bigramcounts:
                    self.bigramcounts[temp_2[i]] += 1
                else:
                    self.bigramcounts[temp_2[i]] = 1

            for i in range(len(temp_3)):
                if temp_3[i] in self.trigramcounts:
                    self.trigramcounts[temp_3[i]] += 1
                else:
                    self.trigramcounts[temp_3[i]] = 1
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        result = 0.0
        try:
            bigram = (trigram[0],trigram[1],)
            result = self.trigramcounts[trigram]/self.bigramcounts[bigram]
        except Exception as e:
            pass
        else:
            pass
        return result

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        result = 0.0
        try:
            unigram = (bigram[0],)
            result = self.bigramcounts[bigram]/self.unigramcounts[unigram]
        except Exception as e:
            pass
        else:
            pass

        return result
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        result = 0.0
        result = self.unigramcounts[unigram]/self.total

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  


        return result

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        p_uni = self.raw_unigram_probability((trigram[2],))
        p_bi = self.raw_bigram_probability((trigram[1],trigram[2],))
        p_tir = self.raw_trigram_probability(trigram)
        result = 0.0
        result = lambda1*p_tir+lambda2*p_bi+lambda3*p_uni
        return result
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        line = get_ngrams(sentence,3)
        log_por = 0.0
        for item in line:
            raw_por = self.smoothed_trigram_probability(item)
            log_por = log_por+math.log2(raw_por)

        return float(log_por)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum_pro = 0.0
        total_words = 0
        for sentence in corpus:
            sen_pro = self.sentence_logprob(sentence)
            sum_pro += sen_pro
            total_words += len(sentence)

        

        l = sum_pro/total_words
        w = 0.0
        w = 2**(-l)

        return w 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_wrong = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            if pp<pp_wrong:
                correct += 1

    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_wrong = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            if pp<pp_wrong:
                correct += 1
        
        return float(correct/total)

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 
    #model = TrigramModel("hw1_data/brown_train.txt")
    # put test code here...
    #part1
    print("Part1")
    print("  uni_gram: "+str(get_ngrams(["natural","language","processing"],1)))
    print("  bi_gram: "+str(get_ngrams(["natural","language","processing"],2)))
    print("  tri_gram: "+str(get_ngrams(["natural","language","processing"],3)))
    #part2
    print("Part2")
    print("  Number of the trigram [('START','START','the')]: "+str(model.trigramcounts[('START','START','the')]))
    print("  Number of the bigram [(START','the')]: "+str(model.bigramcounts[('START','the')]))
    print("  Number of the unigram [('the',)]: "+str(model.unigramcounts[('the',)]))
    #part3
    print("Part3")
    print("  raw probability of (('the',)): "+str(model.raw_unigram_probability(('the',))))   
    print("  raw probability of (('START','the')): "+str(model.raw_bigram_probability(('START','the'))))   
    print("  raw probability of (('START','START','the')): "+str(model.raw_trigram_probability(('START','START','the'))))
    #part4
    print("Part4")
    print("  smoothed probability of (('START','START','the')): "+str(model.smoothed_trigram_probability(('START','START','the'))))
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 


    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # dev_corpus = corpus_reader("hw1_data/brown_test.txt", model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Part6")
    print("  perplexity of test data on training data: "+str(pp))

    dev_corpus_train = corpus_reader(sys.argv[1], model.lexicon)
    pp_train = model.perplexity(dev_corpus_train)
    print("  perplexity of training data on itself: "+str(pp_train))

    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    print("Part7")
    print("  prediction accuracy: "+str(acc))

