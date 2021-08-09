import matplotlib.pyplot as plt
import numpy as np

from itertools import chain
from collections import Counter, defaultdict
#from helpers import show_model, Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from collections import namedtuple

data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)

def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.
    
    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    # TODO: Finish this function!
    
    # Initialize pair_count_dict as dictionary
    pair_count_dict = {}
    
    # Create the pair_count_dict with tags as keys and value as empty dict
    for tag in sequences_A:
        pair_count_dict[tag]={}
    
    # Build the pair_count_dict in the form {tag1:{word1:count,word2:count},tag2:{word1:count,word2:count}}
    for word,tag in sequences_B.stream():
        if tag in pair_count_dict:
            if word in pair_count_dict[tag]:
                pair_count_dict[tag][word] += 1
            else:
                pair_count_dict[tag][word] = 1

    return pair_count_dict


# Calculate C(t_i, w_i)
emission_counts = pair_counts(data.training_set.tagset,data.training_set)


# Create a lookup table mfc_table where mfc_table[word] contains the tag label most frequently assigned to that word
from collections import namedtuple

FakeState = namedtuple("FakeState", "name")

class MFCTagger:
    # NOTE: You should not need to modify this class or any of its methods
    missing = FakeState(name="<MISSING>")
    
    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})
        
    def viterbi(self, seq):
        """This method simplifies predictions by matching the Pomegranate viterbi() interface"""
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in seq] + ["<end>"]))


# TODO: calculate the frequency of each tag being assigned to each word (hint: similar, but not
# the same as the emission probabilities) and use it to fill the mfc_table

word_counts = pair_counts(data.training_set.tagset,data.training_set)

# Create an empty dictionary
mfc_table = {}
# Create an empty list
word_list=[]
# Create a word_list with and element will be in the form ('word',count,'tag')
for key,inner_dict in word_counts.items():
    for kkey,vval in inner_dict.items():
        word_list.append((kkey,vval,key))

# Sort the word_list in the descending order of the word count so that highest fequencies appear first 
word_list = sorted(word_list, key=lambda x:x[1],reverse=True)

# create the mfc_table dictionary in the form {'word':'tag'}. The word with highest frequency appear as the value.
for vocb in data.training_set.vocab:
    mfc_table[vocb]= [word for word in word_list if vocb==word[0]][0][2]
    
# DO NOT MODIFY BELOW THIS LINE
mfc_model = MFCTagger(mfc_table) # Create a Most Frequent Class tagger instance

def replace_unknown(sequence):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    """X should be a 1-D sequence of observations for the model to predict"""
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]  # do not show the start/end state predictions

def accuracy(X, Y, model):
    """Calculate the prediction accuracy by using the model to decode each sequence
    in the input X and comparing the prediction with the true labels in Y.
    
    The X should be an array whose first dimension is the number of sentences to test,
    and each element of the array should be an iterable of the words in the sequence.
    The arrays X and Y should have the exact same shape.
    
    X = [("See", "Spot", "run"), ("Run", "Spot", "run", "fast"), ...]
    Y = [(), (), ...]
    """
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        
        # The model.viterbi call in simplify_decoding will return None if the HMM
        # raises an error (for example, if a test sentence contains a word that
        # is out of vocabulary for the training set). Any exception counts the
        # full sentence as an error (which makes this a conservative estimate).
        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions

def unigram_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequence list that
    counts the number of occurrences of the value in the sequences list. The sequences
    collection should be a 2-dimensional array.
    
    For example, if the tag NOUN appears 275558 times over all the input sequences,
    then you should return a dictionary such that your_unigram_counts[NOUN] == 275558.
    """
    # TODO: Finish this function!
    # Initialize pair_count_dict as dictionary
    unigram_count_dict={}
    
    # Create the pair_count_dict with tags as keys and value as empty dict
    for tag in sequences.tagset:
        unigram_count_dict[tag]=0
        
    for word,tag in sequences.stream():
        if tag in unigram_count_dict:
            unigram_count_dict[tag] += 1

    return unigram_count_dict

# TODO: call unigram_counts with a list of tag sequences from the training set
tag_unigrams = unigram_counts(data.training_set)



def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.
    
    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """

    # TODO: Finish this function!
    # Initialize the bigram dict
    bigram_dict = {}
    
    # Create the bigran_dict with tags as keys as tuple [(NOUN, VERB)]
    for tag in data.training_set.tagset:
        temp_dict = {(tag,x):0 for x in data.training_set.tagset}
        bigram_dict.update(temp_dict)
    
    # Update the value of the dict with the tuple count from the sequence 
    for tag_list in data.training_set.Y:
        tag_len = len(tag_list)
        tag_counter = 0
        while tag_len > tag_counter:
            try:
                key = (tag_list[tag_counter],(tag_list[tag_counter+1]))
                if key in bigram_dict:
                       bigram_dict[key] += 1
            except:
                pass

            tag_counter += 1
    
    return bigram_dict


# TODO: call bigram_counts with a list of tag sequences from the training set
tag_bigrams = bigram_counts(data.training_set)


def starting_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.
    
    For example, if 8093 sequences start with NOUN, then you should return a
    dictionary such that your_starting_counts[NOUN] == 8093
    """
    # TODO: Finish this function!
    # Initialize the starting_dict
    starting_dict = {}
    
    # Create the starting_dict with tags as key
    for tag in data.training_set.tagset:
        starting_dict[tag] = 0
        
    # Update the value of the starting dict 
    for tag_list in data.training_set.Y:
        tag = tag_list[0]
        if tag in starting_dict:
            starting_dict[tag] += 1
    
    return starting_dict

# TODO: Calculate the count of each tag starting a sequence
tag_starts = starting_counts(data.training_set)


def ending_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.
    
    For example, if 18 sequences end with DET, then you should return a
    dictionary such that your_starting_counts[DET] == 18
    """
    # TODO: Finish this function!
    # Initialize the ending_dict
    ending_dict = {}
    
    # Create the starting_dict with tags as key
    for tag in data.training_set.tagset:
        ending_dict[tag] = 0
        
    # Update the value of the starting dict 
    for tag_list in data.training_set.Y:
        tag = tag_list[-1]
        if tag in ending_dict:
            ending_dict[tag] += 1
    
    return ending_dict

# TODO: Calculate the count of each tag ending a sequence
tag_ends = ending_counts(data.training_set)


basic_model = HiddenMarkovModel(name="base-hmm-tagger")

# TODO: create states with emission probability distributions P(word | tag) and add to the model
# (Hint: you may need to loop & create/add new states)
states={}
for tag,words in emission_counts.items():
    total_tag_cnt = tag_unigrams[tag]
    tag_emissions={}
    for word,word_cnt in words.items():
        tag_emissions[word] = word_cnt/total_tag_cnt
    tag_state = State(DiscreteDistribution(tag_emissions),name=tag)
    states[tag]=tag_state
    basic_model.add_states(tag_state)


# TODO: add edges between states for the observed transition frequencies P(tag_i | tag_i-1)
# (Hint: you may need to loop & add transitions
for tag,tag_cnt in tag_starts.items():
    basic_model.add_transition(basic_model.start,states[tag],tag_cnt/sum(tag_starts.values()))
    
for tag,tag_cnt in tag_bigrams.items():
    basic_model.add_transition(states[tag[0]],states[tag[1]],tag_cnt/tag_unigrams[tag[0]])
    
for tag,tag_cnt in tag_ends.items():
    basic_model.add_transition(states[tag],basic_model.end,tag_cnt/sum(tag_ends.values()))



# NOTE: YOU SHOULD NOT NEED TO MODIFY ANYTHING BELOW THIS LINE
# finalize the model
basic_model.bake()




