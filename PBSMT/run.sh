# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

#
# Data preprocessing configuration
#

N_MONO=10000000  # number of monolingual sentences for each language
N_THREADS=48     # number of threads in data preprocessing
SRC=bora           # source language
TGT=es           # target language


#
# Initialize Moses and data paths
#
# main paths
UMT_PATH=$PWD
ORIGIN_DATA_PATH=$PWD/../augmented_tokenized_data
ORIGIN_EMBEDDING_PATH=$PWD/../XLM/emb
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para
EMB_PATH=$DATA_PATH/embeddings

#create paths
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH
mkdir -p $EMB_PATH

# moses
MOSES_PATH=/project/statnlp/changhao/ubuntu-16.04  # PATH_WHERE_YOU_INSTALLED_MOSES
TOKENIZER=$MOSES_PATH/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES_PATH/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES_PATH/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES_PATH/scripts/tokenizer/remove-non-printing-char.perl
TRAIN_TRUECASER=$MOSES_PATH/scripts/recaser/train-truecaser.perl
TRUECASER=$MOSES_PATH/scripts/recaser/truecase.perl
DETRUECASER=$MOSES_PATH/scripts/recaser/detruecase.perl
TRAIN_LM=$MOSES_PATH/bin/lmplz
TRAIN_MODEL=$MOSES_PATH/scripts/training/train-model.perl
MULTIBLEU=$MOSES_PATH/scripts/generic/multi-bleu.perl
MOSES_BIN=$MOSES_PATH/bin/moses

# training directory
TRAIN_DIR=$PWD/moses_train_$SRC-$TGT

# MUSE path
MUSE_PATH=$PWD/MUSE

# files full paths
SRC_TOK=$MONO_PATH/all.$SRC.tok
TGT_TOK=$MONO_PATH/all.$TGT.tok
SRC_TRUE=$MONO_PATH/all.$SRC.true
TGT_TRUE=$MONO_PATH/all.$TGT.true
SRC_VALID=$PARA_PATH/all.$SRC.valid
TGT_VALID=$PARA_PATH/all.$TGT.valid
SRC_TEST=$PARA_PATH/all.$SRC.test
TGT_TEST=$PARA_PATH/all.$TGT.test
SRC_TRUECASER=$DATA_PATH/$SRC.truecaser
TGT_TRUECASER=$DATA_PATH/$TGT.truecaser
SRC_LM_ARPA=$DATA_PATH/$SRC.lm.arpa
TGT_LM_ARPA=$DATA_PATH/$TGT.lm.arpa
SRC_LM_BLM=$DATA_PATH/$SRC.lm.blm
TGT_LM_BLM=$DATA_PATH/$TGT.lm.blm

# copy data
cat $ORIGIN_DATA_PATH/$SRC.train >> $SRC_TOK
cat $ORIGIN_DATA_PATH/$TGT.train >> $TGT_TOK
cat $ORIGIN_DATA_PATH/$SRC.valid >> $SRC_VALID
cat $ORIGIN_DATA_PATH/$TGT.valid >> $TGT_VALID
cat $ORIGIN_DATA_PATH/$SRC.test >> $SRC_TEST
cat $ORIGIN_DATA_PATH/$TGT.test >> $TGT_TEST
cat ORIGIN_EMBEDDING_PATH/$SRC.emb.vec >> $EMB_PATH/$SRC.emb.vec
cat ORIGIN_EMBEDDING_PATH/$TGT.emb.vec >> $EMB_PATH/$TGT.emb.vec

#
# Download and install tools
#

# Check Moses files
if ! [[ -f "$TOKENIZER" && -f "$NORM_PUNC" && -f "$INPUT_FROM_SGM" && -f "$REM_NON_PRINT_CHAR" && -f "$TRAIN_TRUECASER" && -f "$TRUECASER" && -f "$DETRUECASER" && -f "$TRAIN_MODEL" ]]; then
  echo "Some Moses files were not found."
  echo "Please update the MOSES variable to the path where you installed Moses."
  exit
fi
if ! [[ -f "$MOSES_BIN" ]]; then
  echo "Couldn't find Moses binary in: $MOSES_BIN"
  echo "Please check your installation."
  exit
fi
if ! [[ -f "$TRAIN_LM" ]]; then
  echo "Couldn't find language model trainer in: $TRAIN_LM"
  echo "Please install KenLM."
  exit
fi

echo "begin downloading muse"

# Download MUSE
#if [ ! -d "$MUSE_PATH" ]; then
#  echo "Cloning MUSE from GitHub repository..."
#  git clone https://github.com/facebookresearch/MUSE.git
#  cd $MUSE_PATH/data/
#  chmod +x get_evaluation.sh
#  ./get_evaluation.sh
#fi
#echo "MUSE found in: $MUSE_PATH"

# Set embedding path
if [ "$SRC" == "es" ]; then EMB_SRC=$EMB_PATH/es.emb.vec; fi
if [ "$SRC" == "bora" ]; then EMB_SRC=$EMB_PATH/bora.emb.vec; fi
if [ "$TGT" == "es" ]; then EMB_TGT=$EMB_PATH/es.emb.vec; fi
if [ "$TGT" == "bora" ]; then EMB_TGT=$EMB_PATH/bora.emb.vec; fi

echo "Pretrained $SRC embeddings found in: $EMB_SRC"
echo "Pretrained $TGT embeddings found in: $EMB_TGT"

# learn truecasers
if ! [[ -f "$SRC_TRUECASER" && -f "$TGT_TRUECASER" ]]; then
  echo "Learning truecasers..."
  $TRAIN_TRUECASER --model $SRC_TRUECASER --corpus $SRC_TOK
  $TRAIN_TRUECASER --model $TGT_TRUECASER --corpus $TGT_TOK
fi
echo "$SRC truecaser in: $SRC_TRUECASER"
echo "$TGT truecaser in: $TGT_TRUECASER"

# truecase data
if ! [[ -f "$SRC_TRUE" && -f "$TGT_TRUE" ]]; then
  echo "Truecsing monolingual data..."
  $TRUECASER --model $SRC_TRUECASER < $SRC_TOK > $SRC_TRUE
  $TRUECASER --model $TGT_TRUECASER < $TGT_TOK > $TGT_TRUE
fi
echo "$SRC monolingual data truecased in: $SRC_TRUE"
echo "$TGT monolingual data truecased in: $TGT_TRUE"

# learn language models
if ! [[ -f "$SRC_LM_ARPA" && -f "$TGT_LM_ARPA" ]]; then
  echo "Learning language models..."
  $TRAIN_LM -o 5 < $SRC_TRUE > $SRC_LM_ARPA
  $TRAIN_LM -o 5 < $TGT_TRUE > $TGT_LM_ARPA
fi
echo "$SRC language model in: $SRC_LM_ARPA"
echo "$TGT language model in: $TGT_LM_ARPA"

# binarize language models
if ! [[ -f "$SRC_LM_BLM" && -f "$TGT_LM_BLM" ]]; then
  echo "Binarizing language models..."
  $MOSES_PATH/bin/build_binary $SRC_LM_ARPA $SRC_LM_BLM
  $MOSES_PATH/bin/build_binary $TGT_LM_ARPA $TGT_LM_BLM
fi
echo "$SRC binarized language model in: $SRC_LM_BLM"
echo "$TGT binarized language model in: $TGT_LM_BLM"

echo "Truecasing valid and test data..."
$TRUECASER --model $SRC_TRUECASER < $SRC_VALID.tok > $SRC_VALID.true
$TRUECASER --model $TGT_TRUECASER < $TGT_VALID.tok > $TGT_VALID.true
$TRUECASER --model $SRC_TRUECASER < $SRC_TEST.tok > $SRC_TEST.true
$TRUECASER --model $TGT_TRUECASER < $TGT_TEST.tok > $TGT_TEST.true


#
# Running MUSE to generate cross-lingual embeddings
#

ALIGNED_EMBEDDINGS_SRC=$MUSE_PATH/alignments/$SRC$TGT-identical_char/vectors-$SRC.pth
ALIGNED_EMBEDDINGS_TGT=$MUSE_PATH/alignments/$SRC$TGT-identical_char/vectors-$TGT.pth

if ! [[ -f "$ALIGNED_EMBEDDINGS_SRC" && -f "$ALIGNED_EMBEDDINGS_TGT" ]]; then
  rm -rf $MUSE_PATH/alignments/
  echo "Aligning embeddings with MUSE..."
  python $MUSE_PATH/unsupervised.py --src_lang $SRC --tgt_lang $TGT \
  --exp_path $MUSE_PATH --exp_name alignments --exp_id $SRC$TGT-identical_char \
  --src_emb $EMB_SRC \
  --tgt_emb $EMB_TGT \
  --dis_most_frequent 20000\
  --epoch_size 1000000\
  --n_refinement 5 --export "pth"
fi
echo "$SRC aligned embeddings: $ALIGNED_EMBEDDINGS_SRC"
echo "$TGT aligned embeddings: $ALIGNED_EMBEDDINGS_TGT"


#
# Generating a phrase-table in an unsupervised way
#

PHRASE_TABLE_PATH=$MUSE_PATH/alignments/$SRC$TGT-identical_char/phrase-table.$SRC-$TGT.gz
if ! [[ -f "$PHRASE_TABLE_PATH" ]]; then
  echo "Generating unsupervised phrase-table"
  python $UMT_PATH/create-phrase-table.py \
  --src_lang $SRC \
  --tgt_lang $TGT \
  --src_emb $ALIGNED_EMBEDDINGS_SRC \
  --tgt_emb $ALIGNED_EMBEDDINGS_TGT \
  --csls 1 \
  --max_rank 200 \
  --max_vocab 300000 \
  --inverse_score 1 \
  --temperature 45 \
  --phrase_table_path ${PHRASE_TABLE_PATH::-3}
fi
echo "Phrase-table location: $PHRASE_TABLE_PATH"


#
# Train Moses on the generated phrase-table
#

rm -rf $TRAIN_DIR
echo "Generating Moses configuration in: $TRAIN_DIR"

echo "Creating default configuration file..."
$TRAIN_MODEL -root-dir $TRAIN_DIR \
 -f $SRC -e $TGT -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
 -external-bin-dir $MOSES_PATH/training-tools -lm 0:5:$TGT_LM_BLM:8 \
 -cores $N_THREADS -max-phrase-length=4 -score-options "--NoLex" -first-step=9 -last-step=9
CONFIG_PATH=$TRAIN_DIR/model/moses.ini

echo "Removing lexical reordering features ..."
mv $TRAIN_DIR/model/moses.ini $TRAIN_DIR/model/moses.ini.bkp
cat $TRAIN_DIR/model/moses.ini.bkp | grep -v LexicalReordering > $TRAIN_DIR/model/moses.ini

echo "Linking phrase-table path..."
ln -sf $PHRASE_TABLE_PATH $TRAIN_DIR/model/phrase-table.gz

echo "Translating test sentences..."
$MOSES_BIN -threads $N_THREADS -f $CONFIG_PATH < $SRC_TEST.true > $TRAIN_DIR/test.$TGT.hyp.true

echo "Detruecasing hypothesis..."
$DETRUECASER < $TRAIN_DIR/test.$TGT.hyp.true > $TRAIN_DIR/test.$TGT.hyp.tok

echo "Evaluating translations..."
$MULTIBLEU $TGT_TEST.true < $TRAIN_DIR/test.$TGT.hyp.true > $TRAIN_DIR/eval.true
$MULTIBLEU $TGT_TEST.tok < $TRAIN_DIR/test.$TGT.hyp.tok > $TRAIN_DIR/eval.tok

cat $TRAIN_DIR/eval.tok

echo "End of training. Experiment is stored in: $TRAIN_DIR"
