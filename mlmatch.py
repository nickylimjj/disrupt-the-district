#!/usr/local/bin/python

import sys, getopt
import string
from twython import Twython
import json
import ml as model

username = ''
try:
    opts, args = getopt.getopt(sys.argv[1:],"hu:",["help","username="])
except getopt.GetoptError:
    print '[usage] mlmatch.py -u <username> '
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print '[usage] mlmatch.py -u <username>'
        sys.exit()
    elif opt in ("-u", "--username"):
        username = arg
    else:
        assert False, "unhandled option"

# input: twitter username
# output: label
print model.ML_model(username)