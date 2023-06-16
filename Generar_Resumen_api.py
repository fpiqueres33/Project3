
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
import spacy
from gensim.models import LdaModel
from rake_nltk import Rake
from gensim.corpora import Dictionary
from collections import Counter
import logging # chequear los resultados de la API

nlp = spacy.load('es_core_news_md')
stop_words = set(stopwords.words('spanish'))


# preguntar por el fichero y leerlo. Encoding utf-8 para evitar problemas con caracteres especiales
def input_path_and_percentile():
    file_path = input("Indicar archivo con ruta completa (formato txt) --> ")
    percentile_input = input("Introduzca el percentil (0-100) para seleccionar las frases principales (por defecto 80) --> ")
    try:
        percentile = int(percentile_input)
        if percentile < 0 or percentile > 100:
            print("El percentil debe estar entre 0 y 100. Se utilizará el valor por defecto de 80.")
            percentile = 80
    except ValueError:
        print("Entrada no válida. Se utilizará el valor por defecto de 80.")
        percentile = 80
    return file_path, percentile

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().replace('\n', '')
    return data

# Usando spaCy se generan las oraciones
def sentence_segment(data):
    nlp = spacy.load('es_core_news_md')  # Selección idioma español tamaño medio
    doc = nlp(data)
    return [sent.text for sent in doc.sents]

# Tokenizar y lematizar. Pasos necesarios para la creación del resumen
def word_tokenization(sentences):
    return [word_tokenize(sentence) for sentence in sentences]

def lemmatization(tokens):
    lemmas = []
    for token_list in tokens:
        doc = nlp(' '.join(token_list))
        lemmas.append([token.lemma_ for token in doc if token.lemma_ not in stop_words and token.lemma_ not in string.punctuation])
    return lemmas

# Modelo básico con TfidVectorizara para la similitud.
def find_similarity(lemmas):
    sentences = [' '.join(lemma) for lemma in lemmas]
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(X, X)
    return similarity_matrix

# búsqueda del tópico en base a lemmas usando LDA y RAKE.
def generate_topics(lemmas, num_topics=3):
    dictionary = Dictionary(lemmas)
    corpus = [dictionary.doc2bow(lemma) for lemma in lemmas]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    topics = lda_model.print_topics()
    return topics

def extract_key_phrases(text, num_phrases=3):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    key_phrases_with_scores = rake.get_ranked_phrases_with_scores()[:num_phrases]
    max_score = max(score for score, _ in key_phrases_with_scores)
    key_phrases_with_scores_normalized = [(score / max_score, phrase) for score, phrase in key_phrases_with_scores]
    return key_phrases_with_scores_normalized

# Generador de resumen por el top de oraciones con mayor similitud.
def find_top_sentences(similarity_matrix, sentences, percentile=80):
    # Sum of sentence similarities
    sum_similarities = np.sum(similarity_matrix, axis=1)

    print(type(sum_similarities))  # Debug line
    print(sum_similarities)  # Debug line

    # Normalize metrics
    total_sum = np.sum(sum_similarities)
    if total_sum != 0:
        sum_similarities = sum_similarities / total_sum

    # Calculate the threshold score at a certain percentile
    threshold_score = np.percentile(sum_similarities, percentile)

    # Get the indexes of the sentences with similarity scores higher than the threshold score
    top_indexes = [index for index in range(len(sentences)) if sum_similarities[index] > threshold_score]

    # Sort the top sentences based on their similarity scores
    top_sentences = sorted([(sentences[index], sum_similarities[index]) for index in top_indexes],
                           key=lambda x: x[1], reverse=True)

    return top_sentences


# Salida adicional con entidades reconocidas.
def named_entity_recognition(text):
    doc = nlp(text)
    named_entities = []

    for ent in doc.ents:
        named_entities.append((ent.text.upper(), ent.label_))

    entity_counts = Counter(named_entities)

    # Show top 10 entities regardless of their label.
    top_entities = entity_counts.most_common(10)

    return top_entities

# Función para el main
def main(data, percentile=80):
    logging.info('Running sentence segmentation...')
    sentences = sentence_segment(data)

    logging.info('Running word tokenization...')
    tokens = word_tokenization(sentences)

    logging.info('Running lemmatization...')
    lemmas = lemmatization(tokens)

    logging.info('Finding similarity...')
    similarity_matrix = find_similarity(lemmas)

    logging.info('Generating LDA topics...')
    topics = generate_topics(lemmas)

    logging.info('Extracting RAKE key phrases...')
    topics2 = extract_key_phrases(data)

    logging.info('Finding top sentences...')
    top_sentences = find_top_sentences(similarity_matrix, sentences, percentile)

    logging.info('Performing NER...')
    named_entities = named_entity_recognition(data)

    # Print statements are replaced with a returned dictionary
    logging.info('Returning summary...')
    return {
        "lda_topics": topics,
        "rake_topics": topics2,
        "top_sentences": top_sentences,
        "named_entities": named_entities
    }
