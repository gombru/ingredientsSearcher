from load_regressions_from_txt import load_regressions_from_txt
from gensim import corpora, models


database_path = '../../../datasets/recipes5k/regression_output/ingredients_Inception_frozen_200_iter_1700/test.txt'
LDA_model_path = '../../../datasets/recipes5k/models/LDA/lda_model_200.model'
output_file_path = '../../../datasets/recipes5k/results/regression2results_200.txt'
num_topics = 200
max_ing_per_topic = 5
max_ing_per_recipe = 20
topic_threshold = float(1)/num_topics * 2
ing_threshold = 0.0015
# Get topics and associated ingredients
ldamodel = models.ldamodel.LdaModel.load(LDA_model_path)
# topics = ldamodel.print_topics(num_topics=num_topics, num_words=20)
#Get dictionary

out_file = open(output_file_path,'w')

database = load_regressions_from_txt(database_path, num_topics)

for id in database: #For each image
    if ',' in id:
        print "Skiping: " + id
        continue

    img_ing = []
    for t,topic_value in enumerate(database[id]): #For each topic
        if topic_value > topic_threshold: #Topic may contribute with an ingredient
            for ing in ldamodel.show_topic(t,max_ing_per_topic): #For each ningredient associated to the topic
                ing_score = topic_value * ing[1] #Topic prob * ing prob = ingredient confidence
                if ing_score > ing_threshold:
                    img_ing.append([ing[0], ing_score])

    # Sort list by ingredient confidence
    img_ing = sorted(img_ing, key=lambda x: -x[1])
    out_file.write(id.strip('\n') + ',')

    print id + ' --  Num ing: ' + str(len(img_ing))

    for i in range(0,max_ing_per_recipe):
        if i >= len(img_ing): break
        out_file.write(img_ing[i][0] + ',')
    out_file.write('\n')

out_file.close()
print 'DONE'

