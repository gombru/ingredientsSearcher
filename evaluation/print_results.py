from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import sys
sys.path.insert(0, '../ingredients_simplification')
sys.path.insert(0, '../aux')

import clean_ingredients
from simplify_ing_list import simplify_ing_list

results_path = '../../../datasets/recipes5k/results/regression2results_200_topth_0.02ingth_0.0015.txt'

file = open(results_path, "r")

font = ImageFont.truetype("FreeMono.ttf", 16, encoding="unic")

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

for line in file:
    try:
        d = line.split(',')
        img = Image.open('../../../datasets/recipes5k/images/'+ d[0])

        # Find image index
        pos = images_paths.index(d[0])
        index = indices[pos]
        # Find GT ingredients
        gt_ing = ingredients[index].split(',')
        print gt_ing
        gt_ing = simplify_ing_list(gt_ing,blacklist,words2use)

        old_size = img.size
        new_size = (old_size[0] + 260, old_size[1])
        new_im = Image.new("RGB", new_size, (255, 255, 255))  ## luckily, this is already black!
        new_im.paste(img, ((0,0)))
        draw = ImageDraw.Draw(new_im)
        # draw.text((x, y),"Sample Text",(r,g,b))

        for t in range(1,len(d) - 1):
            draw.text((old_size[0] + 10 , t*20),d[t],(0,0,0),font=font)

        draw.text((old_size[0] + 10 + 100, 5), 'GT', (10, 254, 10), font=font)

        for t in range(0,len(gt_ing)):
            draw.text((old_size[0] + 10  + 100 , (t+1)*20),gt_ing[t],(10, 254, 10),font=font)

        if not os.path.exists('../../../datasets/recipes5k/results_img/'+ d[0].split('/')[0]):
            os.makedirs('../../../datasets/recipes5k/results_img/'+ d[0].split('/')[0])
        new_im.save('../../../datasets/recipes5k/results_img/'+d[0])

    except:
        print "ERROR"
        continue

print 'DONE'