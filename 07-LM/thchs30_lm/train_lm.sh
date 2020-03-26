thchs=/users/liuli/database/thchs30

#lexicon=$thchs/data_thchs30/lm_word/lexicon.txt
#cat $lexicon | grep -v "<s>" | grep -v "</s>" | grep -v "#" | grep -v "SIL" | awk '{print $1}' > words.txt

train=$thchs/doc/trans/train.word.txt
test=$thchs/doc/trans/test.word.txt

cat $train | awk '{$1="";print $0}' | sort -u -o train.txt
cat $test | awk '{$1="";print $0}' | sort -u -o test.txt

ngram-count -text train.txt -order 3 -write train.txt.count
ngram-count -read train.txt.count -order 3 -lm myLM.lm -interpolate -kndiscount
