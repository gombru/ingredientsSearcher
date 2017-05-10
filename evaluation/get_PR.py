import os
import sys
sys.path.insert(0, '../ingredients_simplification')
sys.path.insert(0, '../aux')

import clean_ingredients
from simplify_ing_list import simplify_ing_list

results_path = '../../../datasets/recipes5k/results/regression2results_200_topth_0.02ingth_5e-05.txt'

def get_PR(results_path):
    #We have to go from image path to the line in the txt to get the label which is the index in the ingredients txt
    split='test'
    ingredients_path = '../../../datasets/recipes5k/annotations/ingredients_Recipes5k.txt'
    images_path = '../../../datasets/recipes5k/annotations/'+split+'_images.txt'
    indices_path = '../../../datasets/recipes5k/annotations/'+split+'_labels.txt'

    blacklist = clean_ingredients.readBlacklist('../ingredients_simplification/blacklist.txt')
    words2use = clean_ingredients.readBaseIngredients('../ingredients_simplification/simplifiedIngredients.txt')

    indices_file = open(indices_path, 'r')
    ingredients_file = open(ingredients_path, 'r')
    images_paths_file = open(images_path, 'r')

    indices=[]
    ingredients=[]
    images_paths=[]

    for line in indices_file:
        indices.append(int(line.strip('\n')))
    for line in ingredients_file:
        ingredients.append(line.strip('\n'))
    for line in images_paths_file:
        images_paths.append(line.strip('\n'))

    file = open(results_path, "r")

    print "Loading data ..."
    print results_path

    tp=0
    fp=0
    fn=0
    skipped = 0

    for line in file:
        print line
        try:
            cur_tp = 0
            cur_fp= 0
            d = line.split(',')
            # Find image index
            pos = images_paths.index(d[0])
            index = indices[pos]

            # Find GT ingredients
            gt_ing = ingredients[index].split(',')
            gt_ing = simplify_ing_list(gt_ing,blacklist,words2use)


            for t in range(1,len(d) - 1):
                found = False
                i = 0
                while not found and i < len(gt_ing):
                    if gt_ing[i] in d[t]:
                        found = True
                        cur_tp += 1
                    i += 1
                if not found: cur_fp += 1

            # fn are missing are ingredients not nfoun
            cur_fn = (len(gt_ing) - 1) - cur_tp

            tp += cur_tp
            fp += cur_fp
            fn += cur_fn

        except:
            print "Error, skipping"
            skipped += 1

    print 'Skipped: ' + str(skipped)
    print 'TP: ' + str(tp) + ' - FP: ' + str(fp) + ' - FN: ' + str(fn)
    P = 100 * float(tp) / (tp + fp)
    R = 100 * float(tp) / (tp + fn)

    print 'Precision: ' + str(P) + ' - Recall: ' + str(R) + ' - F-score: ' + str(2 * ((P*R) / (P+R)))


    print 'DONE'

get_PR(results_path)