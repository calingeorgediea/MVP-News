import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.ldamulticore import LdaMulticore

# Load the CSV file with tokenized content
df = pd.read_csv("out.csv")

# Create a list of tokenized documents
tokenized_docs = [doc.split() for doc in df["processed_content"].values]

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(tokenized_docs)

# Filter out tokens that appear in less than 5 documents or more than 50% of the documents
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Create a bag-of-words representation of the documents
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Set number of topics
num_topics = 10

# Build the LDA model
lda_model = LdaMulticore(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         workers=3,  # Adjust based on your system
                         passes=100)  # Number of passes through the corpus

# Print the top 5 terms for each topic
for idx, topic in lda_model.print_topics(-1):
    terms = topic.split("+")
    terms = [term.split("*")[1].strip().replace('"', '') for term in terms][:5]
    print("Topic {}: {}".format(idx, ", ".join(terms)))

# Save the model if needed
# lda_model.save("lda_model")
