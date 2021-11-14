import json

# Opening JSON file
f = open('data/trainRecipes.json', 'r', encoding='UTF-8')

for l in f:
    print(eval(l))

# Closing file
f.close()
