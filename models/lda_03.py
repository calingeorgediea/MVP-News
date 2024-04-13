import pandas as pd
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel

# Load the CSV file with tokenized content
df = pd.read_csv("tokenized_output.csv")

# Assuming you have a column "category" in your CSV that indicates the category of each headline
categories = df["category"].unique()

for category in categories:
    print(f"Training LDA for category: {category}")
    
    # Filter the dataframe for the current category
    category_df = df[df["category"] == category]
    
    # Create a list of tokenized documents for this category
    tokenized_docs = [doc.split() for doc in category_df["tokens"].values]
    
    # Check if there are no tokenized documents for this category
    if not tokenized_docs:
        print(f"No documents for category: {category}. Skipping...")
        continue
    
    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(tokenized_docs)
    
    # Filter out tokens that appear in less than 5 documents or more than 50% of the documents
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    # Create a bag-of-words representation of the documents
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    
    # Function to compute coherence score for a given number of topics
    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
        coherence_values = []
        model_list = []
        
        for num_topics in range(start, limit, step):
            model = LdaMulticore(corpus=corpus,
                                 id2word=dictionary,
                                 num_topics=num_topics,
                                 workers=3,  # Adjust based on your system
                                 passes=100)  # Number of passes through the corpus
            model_list.append(model)
            coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())
        
        return model_list, coherence_values
    
    # Set the range of topics to try
    start_topics = 2
    limit_topics = 20
    step_topics = 1
    
    try:
        # Compute coherence values for different number of topics
        model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=tokenized_docs,
                                                                start=start_topics, limit=limit_topics, step=step_topics)
        
        # Find the optimal number of topics with the highest coherence value
        optimal_num_topics = start_topics + coherence_values.index(max(coherence_values))
        print("Optimal number of topics:", optimal_num_topics)
        
        # Build the LDA model with the optimal number of topics
        lda_model = model_list[coherence_values.index(max(coherence_values))]
        
        # Print the top terms for each topic
        for idx, topic in lda_model.print_topics(-1):
            terms = topic.split("+")
            terms = [term.split("*")[1].strip().replace('"', '') for term in terms][:10]
            print("Topic {}: {}".format(idx, ", ".join(terms)))
        
        # Save the model if needed for each category
        # lda_model.save(f"lda_model_{category}.model")
    
    except ValueError as e:
        print(f"Error: {e}. Skipping...")
        continue
