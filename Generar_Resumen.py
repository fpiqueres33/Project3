import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
import spacy
from gensim.models import LdaModel  # para testear los topics (desordenada la frase)
from rake_nltk import Rake # para testear los tópicos (ordenada la frase)
from gensim.corpora import Dictionary
from collections import Counter

nlp = spacy.load('es_core_news_md')
stop_words = set(stopwords.words('spanish'))


def input_path():
    file_path = input("Indicar archivo con ruta completa (formato txt) ")
    return file_path

# Función de lectura de archivo. Codificacion UTF-8 para mejorar la lectura de caracteres especiales.
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().replace('\n', '')
    return data

# segmentar oraciones, idioma español.
def sentence_segment(data):
    return sent_tokenize(data, language='spanish')

# Toekenización de las oraciones
def word_tokenization(sentences):
    return [word_tokenize(sentence) for sentence in sentences]

# Generar lemmas de los tokens, filtrando stopwords
def lemmatization(tokens):
    lemmas = []
    for token_list in tokens:
        doc = nlp(' '.join(token_list))
        lemmas.append([token.lemma_ for token in doc if token.lemma_ not in stop_words and token.lemma_ not in string.punctuation])
    return lemmas

# Similitud de los lemas. Se utiliza Tfidf, como primer modelo.
def find_similarity(lemmas):
    # Convertir las listas de lemmas en strings
    sentences = [' '.join(lemma) for lemma in lemmas]

    # Crear matriz TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Calcular la matriz de similitud del coseno
    similarity_matrix = cosine_similarity(X, X)
    return similarity_matrix

# Generación de tópicos, modelo lda como alternativa 1 de gensim.
def generate_topics(lemmas, num_topics=3):
    dictionary = Dictionary(lemmas)
    corpus = [dictionary.doc2bow(lemma) for lemma in lemmas]

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    topics = lda_model.print_topics()

    return topics

# Generación de tópicos, modelo Rake como alternativa 2. Se añade métrica de similitud
def extract_key_phrases(text, num_phrases=3):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    key_phrases_with_scores = rake.get_ranked_phrases_with_scores()[:num_phrases]

    # Normalizar los scores
    max_score = max(score for score, _ in key_phrases_with_scores)
    key_phrases_with_scores_normalized = [(score / max_score, phrase) for score, phrase in key_phrases_with_scores]

    return key_phrases_with_scores_normalized


# Generador de resumen encontrando las oraciones con mayor similitud, 10 por defecto.
def find_top_sentences(similarity_matrix, sentences, top_n=10):
    # Sumar las similitudes de las frases
    sum_similarities = np.sum(similarity_matrix, axis=1)

    # Normalizar los puntajes para que sumen 1
    sum_similarities = sum_similarities / np.sum(sum_similarities)

    # Obtener los índices de las frases más similares
    top_indexes = np.argsort(sum_similarities)[::-1][:top_n]

    # Obtener las frases más similares y sus puntaciones
    top_sentences = [(sentences[index], sum_similarities[index]) for index in top_indexes]

    return top_sentences


# Extra. Localizar entities para mejorar la comprensión del texto.
def named_entity_recognition(text):
    doc = nlp(text)
    named_entities = []

    for sent in doc.sents:
        for ent in sent.ents:
            # Filtrar las entidades en el caso de inicio de las oraciones. Lo pasamos a MAYUSCULAS
            # para normalizar el texto.
            if sent.text.find(ent.text) != 0 or ent.label_ in ['PERSON', 'GPE', 'ORG']:
                named_entities.append((ent.text.upper(), ent.label_))

    # Contador de entiades por etiqueta.
    entity_counts = Counter(named_entities)

    # Mostrar top 3 de entidades por etiqueta.
    top_entities = {}
    for entity, count in entity_counts.most_common():
        text, label = entity
        if label not in top_entities:
            top_entities[label] = []
        if len(top_entities[label]) < 3:
            top_entities[label].append((text, count))

    return top_entities

# ejecutable main para resultados
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
    for label, entities in named_entities.items():
        print(f"Etiqueta: {label}")
        for entity, count in entities:
            print(f"Entidad: {entity}, Num Ocurrencias: {count}")

