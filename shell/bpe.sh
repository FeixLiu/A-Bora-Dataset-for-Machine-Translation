# Get BPE for Bora
OUTPATH=data/processed/XLM_bora/30k
FASTBPE=tools/fastBPE/fast

# create output path
mkdir -p $OUTPATH

# learn bpe codes on the training set (or only use a subset of it)
$FASTBPE learnbpe 30000 data/bora/bora.train > $OUTPATH/codes

# apply BPE
$FASTBPE applybpe $OUTPATH/train.bora data/bora/bora.train $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/valid.bora data/bora/bora.valid $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/test.bora data/bora/bora.test $OUTPATH/codes

# post-BPE vocabulary
cat $OUTPATH/train.bora | $FASTBPE getvocab - > $OUTPATH/vocab

# binarize the data
python preprocess.py $OUTPATH/vocab $OUTPATH/train.bora
python preprocess.py $OUTPATH/vocab $OUTPATH/valid.bora
python preprocess.py $OUTPATH/vocab $OUTPATH/test.bora

# Get BPE for Spanish
OUTPATH=data/processed/XLM_es/30k
FASTBPE=tools/fastBPE/fast

# create output path
mkdir -p $OUTPATH

# learn bpe codes on the training set (or only use a subset of it)
$FASTBPE learnbpe 30000 data/es/es.train > $OUTPATH/codes

# apply BPE
$FASTBPE applybpe $OUTPATH/train.es data/es/es.train $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/valid.es data/es/es.valid $OUTPATH/codes
$FASTBPE applybpe $OUTPATH/test.es data/es/es.test $OUTPATH/codes

# post-BPE vocabulary
cat $OUTPATH/train.es | $FASTBPE getvocab - > $OUTPATH/vocab

# binarize the data
python preprocess.py $OUTPATH/vocab $OUTPATH/train.es
python preprocess.py $OUTPATH/vocab $OUTPATH/valid.es
python preprocess.py $OUTPATH/vocab $OUTPATH/test.es

# Get BPE for Bora-Spanish pair
OUTPATH=data/processed/XLM_bora/30k
FASTBPE=tools/fastBPE/fast

# create output path
mkdir -p $OUTPATH

# learn bpe codes on the training set (or only use a subset of it)
$FASTBPE learnbpe 30000 data/bora_es/bora_es.train > $OUTPATH/codes

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
