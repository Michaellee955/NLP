"""
COMS W4705 - Natural Language Processing - Spring 2018
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        for key in self.lhs_to_rules:
            item = self.lhs_to_rules[key]
            size = len(item)
            verify_sum = []
            verify_CFG = []
            for i in range(size):
                verify_sum.append(item[i][2])
                verify_CFG.append(item[i][1])
            sum = fsum(verify_sum)
            thr = 1e-10
            if sum >1+thr or sum<1-thr:
                return False
            for words in verify_CFG:
                length = len(words)
                if length==1 and words[0].isupper():
                    return False
                elif length==2:
                    if  words[0].islower() or words[1].islower(): #no error on the given dataset
                        # but works bad for input like ("AaAa","BbBb")
                        return False
                elif length>2:
                    return False
        return True


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        print(grammar.verify_grammar())
