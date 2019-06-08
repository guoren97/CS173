import json
import csv
from bs4 import BeautifulSoup as bs



lines = open("News_Category_Dataset_v2.json")
writer = csv.writer(open("News_Category_Dataset_v5.csv", "w"))
writer.writerow(["CATEGORY", "TITLE", "link", "short_description"])

category = ['ENTERTAINMENT', 'POLITICS', 'WELLNESS', 'TRAVEL', 'STYLE & BEAUTY', 'PARENTING', 'HEALTHY LIVING', 'BLACK VOICES']
#category = ['ENTERTAINMENT', 'POLITICS', 'WELLNESS', 'TRAVEL']
category_count = [0, 0, 0, 0, 0, 0, 0, 0]

while 1:
    line = lines.readline()
    if not line:
        break
    distros_dict = json.loads(line)
    if str(distros_dict['category']) in category:
        category_count[category.index(str(distros_dict['category']))] = category_count[category.index(str(distros_dict['category']))]+1
        if category_count[category.index(str(distros_dict['category']))] < 6000:
            writer.writerow([distros_dict['category'], distros_dict['headline'], distros_dict['link'], distros_dict['short_description']])