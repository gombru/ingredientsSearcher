# Trains and saves an LDA model with the given text files.

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import string
import numpy as np
import sys
sys.path.insert(0, '../ingredients_simplification')
import clean_ingredients


whitelist = string.letters + string.digits + ' ' + ','
text_data_path = '../../../datasets/recipes5k/aux_annotations/lda_train_ingredients.txt'
model_path = '../../../datasets/recipes5k/models/LDA/lda_model_200.model'
blacklist = clean_ingredients.readBlacklist('../ingredients_simplification/blacklist.txt')
words2use = clean_ingredients.readBaseIngredients('../ingredients_simplification/simplifiedIngredients.txt')

num_topics = 200
threads = 8
passes = 20

#Initialize Tokenizer
# tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
# en_stop = get_stop_words('en')
# # add own stop words
# for w in words2filter:
#     en_stop.append(w)
# Create p_stemmer of class PorterStemmer
# p_stemmer = PorterStemmer()

posts_text = []
texts = [] #List of lists of tokens


file = open(text_data_path, "r")
for line in file:
    filtered_text = ""
    # Keep only letters and numbers
    for char in line:
        if char in whitelist:
            filtered_text += char
    posts_text.append(filtered_text.decode('utf-8').lower())


print "Number of posts: " + str(len(posts_text))

print "Creating tokens"
c= 0

for t in posts_text:

    c += 1
    if c % 10000 == 0:
        print c

    try:
        t = t.lower()
        # tokens = tokenizer.tokenize(t)
        tokens = t.split(',')

        for tok in range(0,len(tokens)):

            # Remove words form blacklist
            ing_parts = tokens[tok].split(' ')
            for b in blacklist:
                if b in ing_parts:
                    pos_b = ing_parts.index(b)
                    ing_parts = ing_parts[:pos_b] + ing_parts[pos_b + 1:]
            tokens[tok] = ' '.join(ing_parts).strip()

            # Simplify ingredients if contained in base_ingredients list
            found = False
            i = 0
            while not found and i < len(words2use):
                if words2use[i] in tokens[tok]:
                    tokens[tok] = words2use[i]
                    found = True
                i += 1

            if not found:
                print "Ignoring ingredient: " + tok
                tokens.remove(tokens[tok])

        # remove stop words from tokens
        # stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem token
        # text = [p_stemmer.stem(i) for i in stopped_tokens]
        # add proceced text to list of lists
        texts.append(tokens)
        # print tokens
    except:
        continue
    #Remove element from list if memory limitation TODO
    #del tweets_text[0]

posts_text = []
# Construct a document-term matrix to understand how frewuently each term occurs within each document
# The Dictionary() function traverses texts, assigning a unique integer id to each unique token while also collecting word counts and relevant statistics.
# To see each token unique integer id, try print(dictionary.token2id)
dictionary = corpora.Dictionary(texts)

# Convert dictionary to a BoW
# The result is a list of vectors equal to the number of documents. Each document containts tumples (term ID, term frequency)
corpus = [dictionary.doc2bow(text) for text in texts]
#Randomize training elements
corpus = np.random.permutation(corpus)
texts = []

# Generate an LDA model
print "Creating LDA model"
# ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20)
ldamodel = models.LdaMulticore(corpus, num_topics=num_topics, id2word = dictionary, passes=passes, workers=threads)
ldamodel.save(model_path)
# Our LDA model is now stored as ldamodel

print(ldamodel.print_topics(num_topics=8, num_words=10))

print "DONE"








