from collections import defaultdict
from math import sqrt

class Document:
    def __init__(self, true_class=None):
        self.true_class = true_class
        self.tokens = defaultdict(lambda: 0)

    def add_tokens(self, token_list):
        for item in token_list:
            self.tokens[item] += 1

    def __repr__(self):
        return f"{self.true_class} {self.tokens}"

def euclidean_distance(d1, d2):
    union = d1.tokens.keys() | d2.tokens.keys()
    dist = sum((d1.tokens[item] - d2.tokens[item]) ** 2 for item in union)
    return dist

def cosine_similarity(d1, d2):
    tokens = set(d1.tokens.keys()).union(d2.tokens.keys())
    dot_product = sum(d1.tokens[token] * d2.tokens[token] for token in tokens)
    norm_d1 = sqrt(sum(val ** 2 for val in d1.tokens.values()))
    norm_d2 = sqrt(sum(val ** 2 for val in d2.tokens.values()))
    if norm_d1 == 0 or norm_d2 == 0:
        return 0.0
    return dot_product / (norm_d1 * norm_d2)


