from Cluster import *
from Document import *
from Loader import convert_to_lowercase, remove_trailing_punct, not_stopword, create_easy_documents
from make_dataset import create_docs, generate_random_words
from sklearn.model_selection import KFold

def classify(clusters, item):
    max_sim = -1
    best = None
    for c in clusters:
        sim = cosine_similarity(c.centroid, item)
        if sim > max_sim:
            max_sim = sim
            best = c
    return best.centroid.true_class

def five_fold_cross_validation(nwords, nelements, pos_lexicon, neg_lexicon):
    pos_docs, neg_docs = create_docs(nelements, nelements, nwords, pos_lexicon, neg_lexicon)
    positive_docs = create_easy_documents(pos_docs, 'pos',
                                          filters=[not_stopword],
                                          transforms=[convert_to_lowercase, remove_trailing_punct])
    negative_docs = create_easy_documents(neg_docs, 'neg',
                                          filters=[not_stopword],
                                          transforms=[convert_to_lowercase, remove_trailing_punct])
    data = positive_docs + negative_docs
    labels = [doc.true_class for doc in data]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    fold = 1
    for train_index, test_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        clusters = k_means(2, ['pos', 'neg'], train_data)

        correct = 0
        for doc in test_data:
            predicted_class = classify(clusters, doc)
            if predicted_class == doc.true_class:
                correct += 1

        accuracy = correct / len(test_data)
        accuracies.append(accuracy)
        print(f"Fold {fold} accuracy: {accuracy:.4f}")
        fold += 1

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average accuracy over 5 folds: {average_accuracy:.4f}")

if __name__ == "__main__":
    nwords = 100  # Words per doc
    nelements = 50  # Docs per class
    lexicon_sizes = [25, 50, 100, 500, 1000]

    for size in lexicon_sizes:
        print(f"\nTesting with lexicon size: {size}")
        pos_lexicon = generate_random_words(size)
        neg_lexicon = generate_random_words(size)
        five_fold_cross_validation(nwords, nelements, pos_lexicon, neg_lexicon)






