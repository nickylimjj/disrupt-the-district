#/bin/bash

declare -a people=(
    "realDonaldTrump"
    "katyperry"
    "BarackObama"
    "taylorswift13"
    "rihanna"
    "TheEllenShow"
    "ladygaga"
    "jtimberlake"
    "Cristiano"
    "jimmyfallon"
    "ddlovato"
    "Oprah"
    "BillGates"
    "edsheeran"
    "pakalupapito"
    "BoredElonMusk"   
    )
for var in ${people[@]}
do
    # pull twitter data
    ./twitter.py -u $var -o "$var".json

    # move to data/
    mkdir data/"$var"
    mv "$var".json data/"$var"/"$var"-tweets.json

    # run personality test if not done before
    if [ ! -f "data/"$var"/"$var"-pers.json" ]; then
        ./index.js $var > "$var"-pers.json
    else
        echo "[*] @"$var" generated before"
        continue
    fi

    mv "$var"-pers.json data/"$var"/"$var"-pers.json
done

echo "tweets and personalities generated successfully."