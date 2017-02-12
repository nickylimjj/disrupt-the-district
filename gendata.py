import sys
import json
from subprocess import call
import os

data = json.loads(open('cities.json').read())
#print(data)

def gendata(city, account):
    os.system("python ./twitter.py -u "+account+" -o "+account+".json")
    os.system("./index.js "+account +" > "+account+"-pers.json")
    os.system("mkdir -p data/"+city+"/")
    os.system("mv "+account+".json data/"+city+"/"+account+"-tweets.json")
    os.system("mv "+account+"-pers.json data/"+city+"/"+account+"-pers.json")
    os.system("touch data/"+city+"/"+account+"-label.txt")
    return

def main(argv):
    for city,twitter_accts in data.items():
        #print city, "=>", twitter_accts.split(",")
        for account in twitter_accts.split(","):
            gendata(city, account)
    return

if __name__ == "__main__":
    main(sys.argv[1:])
