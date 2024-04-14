import os
import sys
import spacy
from gensim import corpora, models
from spacy.lang.ro.stop_words import STOP_WORDS
import pandas as pd
nlp = spacy.load("ro_core_news_lg")
def text_pipeline(data_file, num_topics=10, passes=15):
    # Load the spaCy model for Romanian
    

    # Load your text data from CSV
    texts = pd.read_csv(data_file)

    # Tokenize, remove stopwords, and get document vectors
    def process_text(text):
        doc = nlp(text)
        processed_text = [token.text.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha]
        return processed_text

    processed_texts = [process_text(text) for text in texts['cleaned_tokens_final']]

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(processed_texts)

    # Create corpus of document vectors
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)

    # Save the topics and their corresponding words to a single file
    with open(f"./output/{os.path.splitext(os.path.basename(data_file))[0]}_topics.txt", 'w') as f:
        for idx, topic in lda_model.print_topics(num_topics):
            words = topic.split('+')
            topic_words = [word.split('*')[1].replace('"', '').strip() for word in words]
            topic_str = ', '.join(topic_words)
            f.write(f"Topic {idx}:\n")
            f.write(f"{topic_str}\n\n")

    print("Topics saved in a single .txt file")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    text_pipeline(input_file)
