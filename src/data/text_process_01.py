import pandas as pd
import re
import stanza
import spacy_stanza

nlp = spacy_stanza.load_pipeline("ro")

def lemmatize_tokens(tokens):
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    return lemmatized_tokens

def remove_ner(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.ent_type_ not in ['MONEY', 'DATE', 'TIME', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'NUMERIC_VALUE', 'PERSON', 'DATETIME']:
            alpha_chars = [char for char in token.text if char.isalpha()]
            cleaned_token = ''.join(alpha_chars)
            if cleaned_token:  
                tokens.append(cleaned_token)
    return tokens

def remove_stopwords(tokens):
    stopwords = spacy.lang.ro.stop_words.STOP_WORDS
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    return filtered_tokens

def preprocess_text(text):
    tokens = remove_ner(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens

def tokenize_text_from_csv(csv_file):
    df = pd.read_csv(csv_file, usecols=['_id', 'category', 'datePublished', 'content'], nrows=10)  # Read only the first 10 rows
    tokenized_data = []
    
    for index, row in df.iterrows():
        content = row['content']
        tokens = preprocess_text(content)
        tokenized_data.append({
            '_id': row['_id'],
            'category': row['category'],
            'datePublished': row['datePublished'],
            'tokens': tokens
        })
    
    tokenized_df = pd.DataFrame(tokenized_data)
    tokenized_df.to_csv("tokenized_output.csv", index=False, columns=['_id', 'category', 'datePublished', 'tokens'])


csv_file_path = "actualitate.csv"
tokenize_text_from_csv(csv_file_path)
