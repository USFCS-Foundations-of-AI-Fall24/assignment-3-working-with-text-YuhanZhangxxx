import string
from Document import *
from Cluster import *
from make_dataset import create_docs

def create_easy_documents(list_of_tokens_list, true_class, filters=None, transforms=None):
    document_list = []
    for tokens in list_of_tokens_list:
        d = Document(true_class=true_class)
        words = tokens

        if transforms:
            for transform in transforms:
                words = [transform(word) for word in words]

        if filters:
            for f in filters:
                words = [word for word in words if f(word)]

        d.add_tokens(words)
        document_list.append(d)
    return document_list

def not_stopword(token):
    return token.lower() not in ['a', 'an', 'the']

def not_cat(token):
    return token.lower() != 'cat'

def remove_trailing_punct(token):
    return token.rstrip(string.punctuation)

def convert_to_lowercase(token):
    return token.lower()

def compute_homogeneity(list_of_clusters, list_of_classes):
    hlist = []
    for cluster in list_of_clusters:
        class_counts = {}
        for doc in cluster.members:
            class_counts[doc.true_class] = class_counts.get(doc.true_class, 0) + 1
        if cluster.members:
            max_class_count = max(class_counts.values())
            homogeneity = max_class_count / len(cluster.members)
        else:
            homogeneity = 0.0
        hlist.append(homogeneity)
    return hlist

def compute_completeness(list_of_clusters, list_of_classes):
    clist = []
    total_docs_per_class = {}

    for cluster in list_of_clusters:
        for doc in cluster.members:
            total_docs_per_class[doc.true_class] = total_docs_per_class.get(doc.true_class, 0) + 1

    for cls in list_of_classes:
        max_docs_in_cluster = 0
        for cluster in list_of_clusters:
            class_count_in_cluster = sum(1 for doc in cluster.members if doc.true_class == cls)
            if class_count_in_cluster > max_docs_in_cluster:
                max_docs_in_cluster = class_count_in_cluster
        completeness = max_docs_in_cluster / total_docs_per_class.get(cls, 1)
        clist.append(completeness)
    return clist

if __name__ == "__main__":
    positive_lexicon = ['good', 'great', 'excellent', 'fantastic', 'positive']
    negative_lexicon = ['bad', 'terrible', 'poor', 'awful', 'negative']

    num_pos_docs = 10
    num_neg_docs = 10
    doc_length = 50  # Adjust as needed

    # Corrected function call using positional arguments
    pos_docs, neg_docs = create_docs(
        num_pos_docs,
        num_neg_docs,
        doc_length,
        positive_lexicon,
        negative_lexicon
    )

    # Process the documents
    positive_docs = create_easy_documents(
        pos_docs,
        'pos',
        filters=[not_stopword],
        transforms=[convert_to_lowercase, remove_trailing_punct]
    )

    negative_docs = create_easy_documents(
        neg_docs,
        'neg',
        filters=[not_stopword],
        transforms=[convert_to_lowercase, remove_trailing_punct]
    )
