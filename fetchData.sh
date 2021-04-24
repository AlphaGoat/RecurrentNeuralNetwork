#!/bin/bash
wget https://kdd.ics.uci.edu/databases/auslan/allsigns.tar.gz
tar -xvzf allsigns.tar.gz -C dataset
rm allsigns.tar.gz
wget https://kdd.ics.uci.edu/databases/JapaneseVowels/ae.test -P dataset
wget https://kdd.ics.uci.edu/databases/JapaneseVowels/ae.train -P dataset
python3 -m dataset.tools
