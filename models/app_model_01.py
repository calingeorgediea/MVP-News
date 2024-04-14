import os
import sys
import spacy
from gensim import corpora, models
from gensim.models import CoherenceModel
from spacy.lang.ro.stop_words import STOP_WORDS
import pandas as pd
nlp = spacy.load("ro_core_news_lg")


def text_pipeline(data_file, passes=15, start=2, limit=20):
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

    # Calculate coherence scores for different number of topics
    coherence_scores = []
    for num_topics in range(start, limit):
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()
        coherence_scores.append((num_topics, coherence_score))

    # Choose the number of topics with the highest coherence score
    best_num_topics, best_coherence_score = max(coherence_scores, key=lambda x: x[1])
    print(f"Best Number of Topics: {best_num_topics} with Coherence Score: {best_coherence_score}")

    # Train the final LDA model with the best number of topics
    lda_model = models.LdaModel(corpus, num_topics=best_num_topics, id2word=dictionary, passes=passes)

    # Save the topics and their corresponding words to a single file
    with open(f"./output/{os.path.splitext(os.path.basename(data_file))[0]}_topics.txt", 'w') as f:
        for idx, topic in lda_model.print_topics(num_topics=best_num_topics):
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
