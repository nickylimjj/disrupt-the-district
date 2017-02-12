# disrupt-the-district
A project on helping people stay relevant in a constantly evolving jobspace.

if any of the scripts do not work, do the following
```
chmod +x <scriptname>
```

This folder contains the following files
* gendata.sh - generate tweet and personality data
    - depends on index.js and twitter.py
* index.js - runs IBM personality test
* twitter.py - this program scraps twitter accounts
    - [usage] ./twitter.py -u <twitter username> -o <outputfilename>
* data/ - contains folders of twitters accounts with tweets and personality, both stored as json.
* ml.py - trains personality data to get suitable location
* mlmatch.py - run `python mlmatch.py -h` for help


This repo ignores
* node_modules/
* data/

