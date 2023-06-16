import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize

def get_text_statistics(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    paragraphs = text.split('\n\n')

    word_counts = [len(word_tokenize(sentence)) for sentence in sentences]

    num_words = len(words)
    num_sentences = len(sentences)
    num_paragraphs = len(paragraphs)
    mean_words_per_sentence = np.mean(word_counts)
    median_words_per_sentence = np.median(word_counts)

    return {
        "num_words": num_words,
        "num_sentences": num_sentences,
        "num_paragraphs": num_paragraphs,
        "mean_words_per_sentence": mean_words_per_sentence,
        "median_words_per_sentence": median_words_per_sentence
    }

def generate_histogram(text, save_path):
    sentences = sent_tokenize(text)
    word_counts = [len(word_tokenize(sentence)) for sentence in sentences]

    plt.hist(word_counts, bins=3)
    plt.xlabel('Word count')
    plt.ylabel('Frequency')
    plt.title('Word count per sentence')

    plt.savefig(save_path)

    return save_path
