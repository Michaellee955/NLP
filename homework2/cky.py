"""
COMS W4705 - Natural Language Processing - Spring 2018
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.{}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        table = {}
        le = len(tokens)
        with open('atis3.pcfg', 'r') as grammar_file:
            grammar = Pcfg(grammar_file)
        for i in range(le):
            tuple = (i, i + 1)
            tuple_word = (tokens[i],)
            table[tuple] = [grammar.rhs_to_rules[tuple_word][x][0] for x in
                            range(len(grammar.rhs_to_rules[tuple_word]))]
        for length in range(2, le + 1):
            for i in range(0, le - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    try:
                        left_length = len(table[(i, k)])
                        for left in range(left_length):
                            right_length = len(table[(k, j)])
                            if left_length != 0 and right_length != 0:
                                for right in range(right_length):
                                    tuple_words = (table[(i, k)][left], table[(k, j)][right])
                                    if (grammar.rhs_to_rules[tuple_words]):
                                        for n in range(len(grammar.rhs_to_rules[tuple_words])):
                                            word = grammar.rhs_to_rules[tuple_words][n][0]
                                            try:
                                                if word not in table[(i, j)]:
                                                    table[(i, j)].append(word)
                                            except:
                                                table[(i, j)] = [word]
                    except:
                        pass
        print(table)
        try:
            if "TOP" in table[(0, le)]:
                return True
            else:
                return False
        except:
            return False
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = {}
        probs = {}
        le = len(tokens)
        with open('atis3.pcfg', 'r') as grammar_file:
            grammar = Pcfg(grammar_file)
        for i in range(le):
            tuple = (i, i + 1)
            tuple_word = (tokens[i],)
            length = len(grammar.rhs_to_rules[tuple_word])
            table[tuple] = {}
            probs[tuple] = {}
            for x in range(length):
                word = grammar.rhs_to_rules[tuple_word][x][0]
                prob_word = grammar.rhs_to_rules[tuple_word][x][2]
                table[tuple][word] = tokens[i]
                probs[tuple][word] = math.log10(prob_word)

        for length in range(2, le + 1):
            for i in range(0, le - length + 1):
                j = i + length
                table[(i, j)] = {}
                probs[(i, j)] = {}
                for k in range(i + 1, j):
                    try:
                        left_length = len(table[(i, k)])
                        right_length = len(table[(k, j)])
                        if left_length != 0 and right_length != 0:
                            left_key = [q for q in table[(i, k)].keys()]
                            right_key = [p for p in table[(k, j)].keys()]
                            for left in range(left_length):
                                for right in range(right_length):
                                    tuple_words = (left_key[left], right_key[right])
                                    if (grammar.rhs_to_rules[tuple_words]):
                                        for n in range(len(grammar.rhs_to_rules[tuple_words])):
                                            word = grammar.rhs_to_rules[tuple_words][n][0]
                                            prob_word = grammar.rhs_to_rules[tuple_words][n][2]
                                            cur_prob = probs[(i, k)][left_key[left]] + probs[k, j][
                                                right_key[right]] + math.log10(
                                                prob_word)
                                            if None == table[(i, j)].get(word):
                                                table[(i, j)][word] = ((left_key[left], i, k), (right_key[right], k, j))
                                                probs[(i, j)][word] = cur_prob
                                            elif cur_prob > probs[(i, j)][word]:
                                                table[(i, j)][word] = ((left_key[left], i, k), (right_key[right], k, j))
                                                probs[(i, j)][word] = cur_prob
                    except:
                        pass
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    length = len(chart[i,j])
    if not chart[i,j][nt][0][0].isupper():
        print(chart[i,j][nt])
        return (nt,chart[i,j][nt])
    else:
        k = chart[i,j][nt][0][2]
        tree_left = get_tree(chart,i,k,chart[i,j][nt][0][0])
        tree_right = get_tree(chart,k,j,chart[i,j][nt][1][0])
        Tree = (nt,tree_left,tree_right)
        return Tree
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        print(get_tree(table, 0, len(toks), grammar.startsymbol))
