#import nltk
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

nlp = spacy.load('es_core_news_md')
stop_words = set(stopwords.words('spanish'))


# preguntar por el fichero y leerlo. Encoding utf-8 para evitar problemas con caracteres especiales
def input_path():
    file_path = input("Indicar archivo con ruta completa (formato txt) --> ")
    return file_path

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
def find_top_sentences(similarity_matrix, sentences):
    # Suma de similitud de las oraciones
    sum_similarities = np.sum(similarity_matrix, axis=1)

    # Normalizar metricas
    sum_similarities = sum_similarities / np.sum(sum_similarities)

    # Calcular la media y multiplicar por 1.75 para elegir el top de las frases más relevantes.
    mean_score = np.mean(sum_similarities) * 1.75

    # Obtener el índice de las frases mas similares
    top_indexes = [index for index in range(len(sentences)) if sum_similarities[index] > mean_score]

    # Obtener las frases más relevantes
    top_sentences = [(sentences[index], sum_similarities[index]) for index in top_indexes]

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
def main():
    file_path = input_path()
    data = read_file(file_path)
    sentences = sentence_segment(data)
    tokens = word_tokenization(sentences)
    lemmas = lemmatization(tokens)
    similarity_matrix = find_similarity(lemmas)
    topics = generate_topics(lemmas)
    topics2 = extract_key_phrases(data)
    top_sentences = find_top_sentences(similarity_matrix, sentences)
    named_entities = named_entity_recognition(data)

    print("Los tópicos propuestos mediante LDA son:")
    for topic in topics:
        print(topic)

    print("\nLos tópicos propuestos mediante RAKE son::")
    for score, phrase in topics2:
        print(f"Frase: {phrase}, - Puntuación normalizada: {score}")

    print("\nResumen generado sobre frases más relevantes:")
    for sentence, score in top_sentences:
        print(f"{sentence} - (puntuación normalizada: {score})")

    print("\nTop de entidades por etiquetas:")
    for entity, count in named_entities:
        print(f"Entidad: {entity[0]}, Etiqueta: {entity[1]}, Num Ocurrencias: {count}")
