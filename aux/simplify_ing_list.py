def simplify_ing_list(img_list, blacklist, words2use):

    for tok in range(0, len(img_list)):

        # Remove words form blacklist
        ing_parts = img_list[tok].split(' ')
        for b in blacklist:
            if b in ing_parts:
                pos_b = ing_parts.index(b)
                ing_parts = ing_parts[:pos_b] + ing_parts[pos_b + 1:]
                img_list[tok] = ' '.join(ing_parts).strip()

        # Simplify ingredients if contained in base_ingredients list
        found = False
        i = 0
        while not found and i < len(words2use):
            if words2use[i] in img_list[tok]:
                img_list[tok] = words2use[i]
                found = True
            i += 1

        if not found:
            print "Ignoring ingredient: " + img_list[tok]

    return img_list