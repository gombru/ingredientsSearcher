# Load trained LDA model and infer topics for unseen text.
# Make the train/val/test splits for CNN regression training
# It also creates the splits train/val/test randomly

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
from random import randint
import string
from joblib import Parallel, delayed
import numpy as np
import sys
sys.path.insert(0, '../ingredients_simplification')
import clean_ingredients

split = 'val'

# Load data and model
ingredients_path = '../../../datasets/recipes5k/annotations/ingredients_Recipes5k.txt'
images_path = '../../../datasets/recipes5k/annotations/'+split+'_images.txt'
indices_path = '../../../datasets/recipes5k/annotations/'+split+'_labels.txt'
model_path = '../../../datasets/recipes5k/models/LDA/lda_model_200.model'

# Create output files
gt_path = '../../../datasets/recipes5k/lda_gt/'+split+'200.txt'

gt_file = open(gt_path, "w")

num_topics = 200

whitelist = string.letters + string.digits + ' ' + ','
blacklist = clean_ingredients.readBlacklist('../ingredients_simplification/blacklist.txt')
words2use = clean_ingredients.readBaseIngredients('../ingredients_simplification/simplifiedIngredients.txt')

ldamodel = models.ldamodel.LdaModel.load(model_path)

topics = ldamodel.print_topics(num_topics=num_topics, num_words=20)
print topics

# Save a txt with the topics and the weights
file = open('topics.txt', 'w')
i = 0
for item in topics:
    file.write(str(i) + " - ")
    file.write("%s\n" % item[1])
    i+=1
file.close()

indices_file = open(indices_path, 'r')
ingredients_file = open(ingredients_path, 'r')
images_paths_file = open(images_path, 'r')
indices=[]
ingredients=[]
images_paths=[]

for line in indices_file:
    indices.append(line)
for line in ingredients_file:
    ingredients.append(line)
for line in images_paths_file:
    images_paths.append(line)

for c in range(0,len(indices)-1):

    index = int(indices[c])
    cur_ingredients = ingredients[index]
    cur_image_path = images_paths[c]
    if ',' in cur_image_path:
        print 'Skiping: ' + cur_image_path
        continue

    #Process ingredients text
    t = cur_ingredients.lower()
    tokens = t.split(',')

    for tok in range(0, len(tokens)):
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
            print "Igredient not found: "
            print tokens[tok]
            # tokens.remove(tokens[tok])

    # Handle stemmer error
    while "aed" in tokens:
        tokens.remove("aed")
        print "aed error"

    try:
        # text = [p_stemmer.stem(i) for i in stopped_tokens]
        bow = ldamodel.id2word.doc2bow(tokens)
        r = ldamodel[bow]
        # print r
    except:
        print "Tokenizer error"
        continue

        # GT for regression
        # Add zeros to topics without score
    topic_probs = ''
    for t in range(0, num_topics):
        assigned = False
        for topic in r:
            if topic[0] == t:
                topic_probs = topic_probs + ',' + str(topic[1])
                assigned = True
                continue
        if not assigned:
            topic_probs = topic_probs + ',' + '0'

    # print id + topic_probs
    # if 'filet_mignon/47_classic_bacon' in cur_image_path:
    #     print 'f'
    gt_file.write(cur_image_path.strip('\n') + topic_probs + '\n')

gt_file.close()

print "Done"
