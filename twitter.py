#!/usr/local/bin/python

import sys, getopt
import string
from twython import Twython
import json


def main(argv):
    username = ''
    outputfile = 'profile.json'

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hu:o:",["help","username=","ofile="])
    except getopt.GetoptError:
        print '[usage] twitter.py -u <username> -o <outputfile>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print '[usage] twitter.py -u <username> -o <outputfile>'
            sys.exit()
        elif opt in ("-u", "--username"):
            username = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        else:
            assert False, "unhandled option"

    # my keys to use twitter API
    key = 'sYrvrdr26JPhw6kTCiGMd0sgl';
    sec_key = 'Zx27S0BZAlGlUu0Dk2w9FFmlbRmcFFLd7RXZu3de6fM2IZp0bw';

    profile = {}
    tweets = []

    twitter = Twython(key, 
    sec_key,oauth_version=2)
    Access_token = twitter.obtain_access_token()
    t = Twython(key, access_token=Access_token)

    user_timeline = t.search(q=username, count=100)
    for tweet in user_timeline['statuses']:
        item = {}
        item["content"] = tweet['text']
        tweets.append(item.copy())

    profile['contentItems'] = tweets

    
    f = open(outputfile, 'w')

    f.write( json.dumps(profile, ensure_ascii=True) )
    f.close()

    print "file created with", "numTweets = ", len(tweets)

if __name__ == "__main__":
    main(sys.argv[1:])