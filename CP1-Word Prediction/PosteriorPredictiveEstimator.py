'''
Summary
-------
Defines a posterior predictive estimator for unigrams

Supports a common API for unigram probability estimators:
* fit
* predict_proba
* score

Examples
--------
>>> word_list = ['dinosaur', 'trex', 'dinosaur', 'stegosaurus']
>>> ppe = PosteriorPredictiveEstimator(Vocabulary(word_list), alpha=2.0)
>>> ppe.fit(word_list)
>>> np.allclose(ppe.predict_proba('dinosaur'), 4.0/10.0)  # wont pass starter
True

>>> ppe.predict_proba('never_seen-before')                # wont pass starter
Traceback (most recent call last):
...
KeyError: 'Word never_seen-before not in the vocabulary'


Resources
---------
See CS 136 course website for problem description and math details
'''

import numpy as np
from Vocabulary import Vocabulary


class PosteriorPredictiveEstimator():
    """
    Posterior Predictive Estimator for unigram probabiliies

    Attributes
    ----------
    vocab : Vocabulary object
    alpha : float, must be greater than or equal to one
            Defines concentration parameter of the symmetric Dirichlet prior

    Internal State Attributes (filled after calling fit)
    -------------------------
    total_count : int
        Total number of words observed in train word list
    count_V : 1D array of size V and type int
        Each entry gives count of corresponding vocab word in train word list
    """

    def __init__(self, vocab, alpha):
        self.vocab = vocab
        self.alpha = float(alpha)

        # Attributes that are adjusted by calls to 'fit'
        self.total_count = 0
        self.count_V = None

    def fit(self, word_list):
        ''' Fit this estimator to provided training data

        Args
        ----
        word_list : list of str
            Each entry is a word that can be looked up in the vocabulary

        Returns
        -------
        None. Internal attributes updated.

        Post Condition
        --------------
        Attributes will updated based on provided word list
        * 1D array count_V is set to the count of each vocabulary word
        * integer total_count is set to the total length of the word list
        '''
        self.total_count = 0
        self.count_V = np.zeros(self.vocab.size)
        for word in word_list:
            self.total_count += 1
            id = self.vocab.get_word_id(word)
            self.count_V[id] += 1

    def predict_proba(self, word):
        ''' Predict probability of a given unigram under this model

        Assumes this word is in the vocabulary

        Args
        ----
        word : string
            Known word that can be looked up in the vocabulary

        Returns
        -------
        proba : float between 0 and 1

        Raises
        ------
        KeyError if the provided word is not in the vocabulary
        '''
        id = self.vocab.get_word_id(word)
        count = self.count_V[id] 

        # Calculate estimated proba of the provided word
        prediction = (count + self.alpha) / (self.total_count
                                            + (self.vocab.size * self.alpha))

        return prediction

    def score(self, word_list):
        ''' Compute the average log probability of words in provided list

        Args
        ----
        word_list : list of str
            Each entry is a word that can be looked up in the vocabulary

        Returns
        -------
        avg_log_proba : float between (-np.inf, 0.0)
        '''
        total_log_proba = 0.0
        for word in word_list:
            total_log_proba += np.log(self.predict_proba(word))
        return total_log_proba / len(word_list)
