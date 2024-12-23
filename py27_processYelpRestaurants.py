import math
import json
import pickle
import random
from gensim import models
from gensim import matutils
from gensim.corpora import Dictionary
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from nltk.tokenize import sent_tokenize
import glob
import argparse
import os
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import numpy as np
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
nltk.download('punkt')  # This downloads the tokenizer
nltk.download('stopwords')  # This downloads the stopwords
nltk.download('punkt_tab')



def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation and numbers
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return tokens

path2files="yelp_dataset_challenge_academic_dataset/"
path2buisness=path2files+"yelp_academic_dataset_business.json"
path2reviews=path2files+"yelp_academic_dataset_review.json"

def main(save_sample, save_categories):
    categories = set([])
    restaurant_ids = set([])
    cat2rid = {}
    rest2rate={}
    rest2revID = {}
    r = 'Restaurants'
    with open (path2buisness, 'r') as f:
        for line in f.readlines():
            business_json = json.loads(line)
            bjc = business_json['categories']
            #cities.add(business_json['city'])
            if r in bjc:
                if len(bjc) > 1:
                    #print(bjc)
                    restaurant_ids.add(business_json['business_id'])
                    categories = set(bjc).union(categories) - set([r])
                    stars = business_json['stars']
                    rest2rate[ business_json['business_id'] ] = stars
                    for cat in bjc:
                        if cat == r:
                            continue
                        if cat in cat2rid:
                            cat2rid[cat].append(business_json['business_id'])
                        else:
                            cat2rid[cat] = [business_json['business_id']]

    print("saving restaurant ratings")
    with open ( 'restaurantIds2ratings.txt', 'w') as f:
        for key in rest2rate:
            f.write( key + " " + str(rest2rate[key]) + "\n")
    #clearing from memory
    rest2rate=None
    with open('data_cat2rid.pickle', 'wb') as f:
        pickle.dump(cat2rid,f)

    with open (path2reviews, 'r') as f:
        for line in f.readlines():
            review_json = json.loads(line)
            if review_json['business_id'] in restaurant_ids:
                if review_json['business_id'] in rest2revID:
                    rest2revID[ review_json['business_id'] ].append(review_json['review_id'])
                else:
                    rest2revID[ review_json['business_id'] ] = [ review_json['review_id'] ]

    with open('data_rest2revID.pickle', 'wb') as f:
        pickle.dump(rest2revID,f)

    nz_count = 0
    valid_cats = []
    for i, cat in enumerate(cat2rid):
        cat_total_reviews = 0
        for rid in cat2rid[cat]:
            #number of reviews for each of restaurants
            if rid in rest2revID:
                cat_total_reviews = cat_total_reviews + len(rest2revID[rid])

        if cat_total_reviews > 30:
            nz_count = nz_count + 1
            valid_cats.append(cat)
            #print( cat, cat_total_reviews)

    #print nz_count, ' non-zero number of reviews in categories out of', len(cat2rid), 'categories')
    #x = range(nz_count)
    print("sampling categories")
    sample_rid2cat={}
    sample_size = 10 #len(valid_cats) # This specifies how many cuisines you would like to save 
                                  # if this process takes too long you can change it to something smaller like 5, 6 ...
    cat_sample = random.sample(valid_cats, sample_size)
    for cat in cat_sample:
        for rid in cat2rid[cat]:
            if rid in rest2revID:
                if rid not in sample_rid2cat:
                    sample_rid2cat[rid] = []
                sample_rid2cat[rid].append(cat)
    #remove from memory
    rest2revID=None
    #    print (len(sample_rid2cat), len(cat2rid), len(valid_cats), len(cat_sample))
    
    print("reading from reviews file...")
    #ensure categories is a directory
    sample_cat2reviews={}
    sample_cat2ratings={}
    num_reviews = 0
    with open (path2reviews, 'r') as f:
        for line in f.readlines():
            review_json = json.loads(line)
            rid = review_json['business_id']
            if rid in sample_rid2cat:
                for rcat in sample_rid2cat [ rid ]:
                    num_reviews = num_reviews + 1
                    if rcat in sample_cat2reviews:
                        sample_cat2reviews [ rcat ].append(review_json['text'])
                        sample_cat2ratings [ rcat ].append( str(review_json['stars']) )
                    else:
                        sample_cat2reviews [ rcat ] = [review_json['text']]
                        sample_cat2ratings [ rcat ] = [ str(review_json['stars']) ]

    if save_categories:
        print("Saving categories...")
        if not os.path.exists("categories"):
            os.makedirs("categories")
        for cat in sample_cat2reviews:
            filename = 'categories/' + cat.replace('/', '-').replace(" ", "_") + ".txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sample_cat2reviews[cat]))

    if save_sample:
        print("sampling restaurant reviews")
        #save sample for restaurant reviews
        sample_size = min(100000, num_reviews)
        rev_sample = random.sample(range(num_reviews), sample_size)
        my_sample_v2 = []
        sample_ratings = []
        sorted_rev_sample = sorted(rev_sample)
        count = 0
        max_bound = 0
        for cat in sample_cat2reviews:
            print(cat)
            new_max_bound = max_bound + len(sample_cat2reviews[cat])
            while count < sample_size and sorted_rev_sample[count] < new_max_bound:
                my_sample_v2.append( sample_cat2reviews[cat][ sorted_rev_sample[count] - max_bound ].replace("\n", " ").strip() )
                sample_ratings.append( sample_cat2ratings[cat][ sorted_rev_sample[count] - max_bound ] )
                count = count + 1
            max_bound = new_max_bound
            #if count in rev_sample:
            #    my_sample.append(rev.replace("\n", " ").strip())
            #count = count + 1

        with open("review_sample_100000.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(my_sample_v2))

        with open("review_ratings_100000.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_ratings))


def sim_matrix():

    # Logging setup
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Check if the 'categories' folder exists
    if not os.path.isdir("categories"):
        print("The 'categories' folder is missing. Run the script with --cuisine first.")
        return

    # Load text data
    text = []
    c_names = []
    cat_list = glob.glob("categories/*")
    cat_size = len(cat_list)

    if cat_size < 1:
        print("No files found in the 'categories' folder.")
        return

    # Sample up to 30 cuisines
    sample_size = min(30, cat_size)
    cat_sample = sorted(random.sample(range(cat_size), sample_size))

    # Load text content from sampled cuisines
    for i, item in enumerate(cat_list):
        if i in cat_sample:
            with open(item, 'r', encoding='utf-8') as f:
                content = f.read().replace("\n", " ").strip()
                if content:
                    text.append(content)
                    c_names.append(os.path.basename(item).replace("_", " ").replace(".txt", ""))

    # Check if any text was loaded
    if not text:
        print("No valid text data found in the 'categories' folder.")
        return

    print(f"Loaded {len(text)} cuisines for processing.")

    # Tokenize text and create dictionary
    tokenized_text = [preprocess_text(doc) for doc in text]

    dictionary = Dictionary(tokenized_text)
    dictionary.save('corpus_dictionary.dict')  # Save dictionary for future use

    # Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_text]

    # Train LDA model
    lda = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10, iterations=500)

    # Save topics to a file
    with open('topic_words.txt', 'w') as f:
        for topic_id in range(lda.num_topics):
            words = lda.show_topic(topic_id, topn=10)
            readable_words = [(word_id, weight) for word_id, weight in words]  # Fix applied here
            f.write(f"Topic {topic_id}: {readable_words}\n")

    # Save topic distributions for each document
    with open('topic_distributions.txt', 'w') as f:
        for doc_id, topics in enumerate(lda.get_document_topics(corpus)):
            f.write(f"Document {doc_id}: {topics}\n")

    # Generate and save pyLDAvis visualization
    vis = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(vis, 'lda_visualization.html')
    print("LDA visualization saved as 'lda_visualization.html'. Open this file in a browser to view the topics.")

    # Compute similarity matrix
    doc_topics = lda.get_document_topics(corpus)
    cuisine_matrix = []
    for i, doc_a in enumerate(doc_topics):
        sim_vecs = []
        for j, doc_b in enumerate(doc_topics):
            if i <= j:
                norm_a = sum(weight_a ** 2 for _, weight_a in doc_a)
                norm_b = sum(weight_b ** 2 for _, weight_b in doc_b)
                norm_a = math.sqrt(norm_a)
                norm_b = math.sqrt(norm_b)
                w_sum = sum(weight_a * weight_b for (topic_a, weight_a) in doc_a
                            for (topic_b, weight_b) in doc_b if topic_a == topic_b)
                denom = norm_a * norm_b
                sim = w_sum / denom if denom > 0 else 0
            else:
                sim = cuisine_matrix[j][i]
            sim_vecs.append(sim)
        cuisine_matrix.append(sim_vecs)

    # Save similarity matrix
    with open('cuisine_sim_matrix.csv', 'w') as f:
        for i_list in cuisine_matrix:
            max_val = max(i_list) if max(i_list) > 0 else 1  # Prevent division by zero
            normalized = [str(val / max_val) for val in i_list]
            f.write(",".join(normalized) + "\n")

    # Save cuisine indices
    with open('cuisine_indices.txt', 'w') as f:
        f.write("\n".join(c_names))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='This program transforms the Yelp data and saves the cuisines in the category directory. It also samples reivews from Yelp. It can also generates a cuisine similarity matrix.')
    
    parser.add_argument('--cuisine', action='store_true',
                       help='Saves a sample (10) of the cuisines to the "categories" directory. For Task 2 and 3 you will experiment with individual cuisines. This option allows you to generate a folder that contains all of the cuisines in the Yelp dataset. You can run this multiple times to generate more samples or if your machine permits you can change a sample parameter in the code.')
    parser.add_argument('--sample', action='store_true',
                       help='Sample a subset of reviews from the yelp dataset which could be useful for Task 1. This will samples upto 100,000 restaurant reviews from 10 cuisines and saves the output in "review_sample_100000.txt", it also saves their corresponding raitings in the "review_ratings_100000.txt" file. You can run this multiple times to get several different samples.')
    parser.add_argument('--matrix', action='store_true',
                       help='Generates the cuisine similarity matrix which is used for Task 2. First we apply topic modeling to a sample (30) of the cuisines in the "categories" folder and measures the cosine similarity of two cuisines from their topic weights. This might take from half-an-hour to several hours time depending on your machine. The number of topics is 20 and the default number of features is 10000.')
    parser.add_argument('--all', action='store_true',
                       help='Does all of the above.')

    
    args = parser.parse_args()        
    if args.all or (args.sample and args.cuisine):
        print("saving sample and cuisine")
        main(True,True)
    elif args.sample:
        print("generating sample")
        main(args.sample, args.cuisine)
    elif args.cuisine:
        print("generating cuisine")
        main(args.sample, args.cuisine)

    if args.matrix or args.all:
        print("generating cuisine matrix")
        sim_matrix()
    #main()

