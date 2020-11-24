# Get BPE for Bora-Spanish pair
OUTPATH=data/processed/XLM_bora_es/60k
FASTBPE=tools/fastBPE/fast

# create output path
mkdir -p $OUTPATH

# learn bpe codes on the training set (or only use a subset of it)
$FASTBPE learnbpe 60000 data/bora_es/bora_es.train > $OUTPATH/codes

# apply BPE
$FASTBPE applybpe $OUTPATH/train.bora_es data/bora_es/bora_es.train $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/valid.bora_es data/bora_es/bora_es.valid $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/test.bora_es data/bora_es/bora_es.test $OUTPATH/codes

# post-BPE vocabulary
cat $OUTPATH/train.bora_es | $FASTBPE getvocab - > $OUTPATH/vocab

# binarize the data
python preprocess.py $OUTPATH/vocab $OUTPATH/train.bora_es
python preprocess.py $OUTPATH/vocab $OUTPATH/valid.bora_es
python preprocess.py $OUTPATH/vocab $OUTPATH/test.bora_es
