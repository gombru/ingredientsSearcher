from load_regressions_from_txt import load_regressions_from_txt
from gensim import corpora, models
from get_PR import get_PR


database_path = '../../../datasets/recipes5k/regression_output/ingredients_Inception_frozen_500_raw_iter_1400/test.txt'
LDA_model_path = '../../../datasets/recipes5k/models/LDA/lda_model_500_raw.model'
num_topics = 500
max_ing_per_topic = 5
max_ing_per_recipe = 20
topic_threshold = 0.005
ing_threshold = 0.0015

output_file_path = '../../../datasets/recipes5k/results/regression2results_500_raw_topth_' + str(topic_threshold) + 'ingth_' + str(ing_threshold) + '.txt'


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
    aux_list = []
    for t,topic_value in enumerate(database[id]): #For each topic
        if topic_value > topic_threshold: #Topic may contribute with an ingredient
            for ing in ldamodel.show_topic(t,max_ing_per_topic): #For each ningredient associated to the topic
                ing_score = topic_value * ing[1] #Topic prob * ing prob = ingredient confidence
                if ing_score > ing_threshold:
                    try:
                        if ing[0] not in aux_list:
                            aux_list.append(ing[0])
                            img_ing.append([ing[0], ing_score])
                        else:
                            index = aux_list.index(ing[0])
                            score = max(img_ing[1][1], ing_score)
                            #score = img_ing[1][1] + ing_score
                            img_ing[index] = [ing[0], score]
                    except:
                        "Failed saving ingredient"

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
print output_file_path

print "Computing P-R:"
get_PR(output_file_path)

