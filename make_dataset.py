import random
import string

def generate_random_words(n):
    words = set()
    while len(words) < n:
        word_length = random.randint(3, 8)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
        words.add(word)
    return list(words)

def create_docs(npos, nneg, length, pos_lexicon, neg_lexicon):
    stopwords = ['a', 'an', 'the']
    pos_docs = []
    neg_docs = []

    for _ in range(npos):
        d = []
        for _ in range(length):
            if random.random() < 0.2:
                word = random.choice(stopwords)
            else:
                word = random.choice(pos_lexicon)
                if random.random() < 0.2:
                    word = word.capitalize()
                if random.random() < 0.3:
                    word += random.choice(string.punctuation)
            d.append(word)
        pos_docs.append(d)

    for _ in range(nneg):
        d = []
        for _ in range(length):
            if random.random() < 0.2:
                word = random.choice(stopwords)
            else:
                word = random.choice(neg_lexicon)
                if random.random() < 0.2:
                    word = word.capitalize()
                if random.random() < 0.3:
                    word += random.choice(string.punctuation)
            d.append(word)
        neg_docs.append(d)

    return pos_docs, neg_docs

if __name__ == "__main__":
    lexicon_sizes = [25, 50, 100, 500, 1000]
    nwords_per_doc = 100
    n_docs_per_class = 100

    for size in lexicon_sizes:
        print(f"\nGenerating datasets with lexicon size: {size}")
        pos_lexicon = generate_random_words(size)
        neg_lexicon = generate_random_words(size)
        pos_docs, neg_docs = create_docs(
            npos=n_docs_per_class,
            nneg=n_docs_per_class,
            length=nwords_per_doc,
            pos_lexicon=pos_lexicon,
            neg_lexicon=neg_lexicon
        )


