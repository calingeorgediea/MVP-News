import spacy
import gensim
from gensim import corpora, models
from spacy.lang.ro.stop_words import STOP_WORDS
import pandas as pd

def text_pipeline(data_file, num_topics=5, passes=15):
    # Load the spaCy model for Romanian
    nlp = spacy.load("ro_core_news_lg")

    # Load your text data from CSV
    texts = pd.read_csv(data_file)

    # Tokenize, remove stopwords, and get document vectors
    def process_text(text):
        doc = nlp(text)
        processed_text = [token.text.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha]
        return processed_text

    processed_texts = [process_text(text) for text in texts['tokens']]

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(processed_texts)

    # Create corpus of document vectors
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)

    # Print the topics and their corresponding words without weights
    for idx, topic in lda_model.print_topics(10):
        words = topic.split('+')
        topic_words = [word.split('*')[1].replace('"', '').strip() for word in words]
        print(f'Topic: {idx}')
        print(f'Words: {topic_words}')
        print()

    # Save the LDA model
    lda_model.save('lda_model_spacy')

# Example Usage:
text_pipeline('tokenized_output.csv')
