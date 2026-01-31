# ---------------------------------------------------
# Automatic Text Summarization using TF-IDF
# ---------------------------------------------------


import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

# ---------------------------------------------------
# Function to preprocess and split text into sentences
# ---------------------------------------------------
def preprocess_text(input_text):
    """
    This function takes raw input text and splits it
    into individual sentences using NLTK.
    """
    sentences = nltk.sent_tokenize(input_text)
    return sentences

# ---------------------------------------------------
# Function to calculate TF-IDF matrix
# ---------------------------------------------------
def calculate_tfidf(sentences):
    """
    This function calculates the TF-IDF values
    for the given list of sentences.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix

# ---------------------------------------------------
# Function to score sentences
# ---------------------------------------------------
def score_sentences(tfidf_matrix):
    """
    This function calculates sentence scores by
    summing TF-IDF values of words in each sentence.
    """
    scores = np.sum(tfidf_matrix.toarray(), axis=1)
    return scores

# ---------------------------------------------------
# Function to generate summary
# ---------------------------------------------------
def generate_summary(sentences, sentence_scores, summary_ratio=0.4):
    """
    This function selects top-ranked sentences
    based on sentence scores and generates summary.
    """
    summary_length = int(len(sentences) * summary_ratio)
    
    top_sentence_indices = np.argsort(sentence_scores)[-summary_length:]
    
    top_sentence_indices = sorted(top_sentence_indices)
    
    # Generate summary text
    summary = " ".join([sentences[i] for i in top_sentence_indices])
    return summary

# ---------------------------------------------------
# Main program
# ---------------------------------------------------
if __name__ == "__main__":

    text = """
    Machine learning based approaches to text summarization were developed to overcome the limitations of traditional statistical and graph-based techniques. Unlike earlier methods that relied on fixed rules or handcrafted features, machine learning techniques attempt to learn summarization patterns directly from data. These approaches treat text summarization as a learning problem, where the system is trained to identify important sentences based on previously seen examples.

In machine learning based summarization, various features are extracted from sentences to determine their importance. These features may include sentence length, position of the sentence in the document, presence of keywords, word frequency, and similarity with the document title. A machine learning model is then trained using these features to classify sentences as important or non-important. Based on the predictions of the model, the most relevant sentences are selected to form the summary. Commonly used algorithms include decision trees, support vector machines, and naïve Bayes classifiers.

One major advantage of machine learning based approaches is their ability to combine multiple features and make better decisions compared to simple statistical methods. By learning from training data, these systems can adapt to different types of documents and summarization requirements. As a result, machine learning techniques often produce more informative and less repetitive summaries. These methods also allow flexibility in adjusting the model according to specific domains such as news articles, academic papers, or technical documents.
    """

    print("========== ORIGINAL TEXT ==========\n")
    print(text)

    sentences = preprocess_text(text)
    print("\nNumber of sentences:", len(sentences))

    tfidf_matrix = calculate_tfidf(sentences)

    sentence_scores = score_sentences(tfidf_matrix)

    print("\n========== SENTENCE SCORES ==========")
    for i, score in enumerate(sentence_scores):
        print(f"Sentence {i+1}: Score = {score:.4f}")

    summary = generate_summary(sentences, sentence_scores)

    print("\n========== GENERATED SUMMARY ==========\n")
    print(summary)
