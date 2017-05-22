recipes_filename = '../../../datasets/recipes5k/annotations/ingredients_Recipes5k.txt'
indices_filename = '../../../datasets/recipes5k/annotations/train_labels.txt'

out_filename = '../../../datasets/recipes5k/aux_annotations/lda_train_ingredients_raw.txt'

recipes =[]
out_recipes=[]

file = open(recipes_filename, "r")
for line in file:
    line = line.rstrip('\n').strip().lower()
    recipes.append(line)
file.close()

print "Recipes read: " + str(len(recipes))

file = open(indices_filename, "r")
for line in file:
    out_recipes.append(recipes[int(line)])
file.close()


print "Resulting recipes: " + str(len(out_recipes))

file = open(out_filename, "w")
for r in out_recipes:
    file.write(r + '\n')
file.close()
