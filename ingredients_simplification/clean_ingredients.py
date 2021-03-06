

def readIngredientsDictionary(file):

    ing2idx = dict()
    count_ing = 0
    with open(file, 'r') as file:
        for line in file:
            line = line.rstrip('\n').split(',')
            for ing in line:
                ing = ing.strip().lower()
                if ing not in ing2idx.keys():
                    ing2idx[ing] = count_ing
                    count_ing += 1

    idx2ing = {v:k for k,v in ing2idx.iteritems()}

    #return ing2idx, idx2ing
    return ing2idx.keys()


def readBlacklist(file):

    blacklist = []
    with open(file, 'r') as file:
        for line in file:
            line = line.rstrip('\n').strip().lower()
            blacklist.append(line)

    return blacklist

def readBaseIngredients(file):

    base_ingredients = []
    with open(file, 'r') as file:
        for line in file:
            line = line.rstrip('\n').split(',')
            for ing in line:
                ing = ing.strip().lower()
                base_ingredients.append(ing)

    return base_ingredients

def buildIngredientsMapping(ingredients, blacklist, base_ingredients=None):

    ing_mapping = dict()
    new_ing = []
    # Iterate over each ingredient
    for ing in ingredients:
        if ing == 'granulated sugar':
            print ing
        old_ing = ing.strip()

        # Clean ingredient name with all blacklist terms
        ing_parts = ing.split()
        for b in blacklist:
            if b in ing_parts:
                pos_b = ing_parts.index(b)
                ing_parts = ing_parts[:pos_b]+ing_parts[pos_b+1:]
        ing = ' '.join(ing_parts).strip()

        # Simplify ingredients if contained in base_ingredients list
        found = False
        i = 0
        while not found and i < len(base_ingredients):
            if base_ingredients[i] in ing:
                ing = base_ingredients[i]
                found = True
            i += 1

        # Found a new basic ingredient
        if ing not in new_ing:
            new_ing.append(ing)
            idx = len(new_ing)-1
        else: # Found a matching with an already existent basic ingredient
            idx = new_ing.index(ing)

        # Insert in mapping
        ing_mapping[old_ing] = idx

    return new_ing, ing_mapping


if __name__ == "__main__":
    #ing2idx, idx2ing = readIngredientsDictionary('ingredients_Recipes5k.txt')
    ingredients = readIngredientsDictionary('../../../datasets/recipes5k/annotations/ingredients_Recipes5k.txt')
    print 'Unique ingredients:',len(ingredients)
    blacklist = readBlacklist('blacklist.txt')
    print 'Blacklist terms:',len(blacklist)
    base_ingredients = readBaseIngredients('baseIngredients.txt')
    print 'Base ingredients:',len(base_ingredients)
    clean_ingredients_list, raw2clean_mapping = buildIngredientsMapping(ingredients, blacklist,
                                                                        base_ingredients=base_ingredients)
    print 'Clean ingredients:',len(clean_ingredients_list)

    file = open('simplifiedIngredients.txt','w')
    for el in clean_ingredients_list: file.write(el + ',')
    file.close()

    #print clean_ingredients_list
