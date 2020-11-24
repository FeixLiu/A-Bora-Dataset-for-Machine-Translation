# Readme
## Introduction
- This is the project for the Boston University 2020 Fall CS591-W1 Toward Universal Natural Language Understanding
- This project mainly collects a dataset for Bora language (a low-resource language). 
- First use TF-IDF and KNN data augmentation methods on the data, then use the dataset to train a language model MLM. With the pre-trained language model, this project will get the embeddings for every word in the vocabulary. Further use the dataset and the embeddings to train two unsupervised machine translation models: NMT and PBMST to verify the performance of the dataset with the Bleu score. 
- The result shows that the BLEU score is higher for the dataset with augmentation than that without augmentation.

## Folder structure
- PBSMT
    - The codes for training unsupervised PBSMT model and translating sentences from source language to target language.
- augmented_tokenized_data
    - The tokenized data for augmented Bora dataset and for Spanish dataset
- data_augmentation
    - The codes for TF-IDF augmentation and word embeddings augmentation
- origin_data
    - The origin data for Bora and Spanish
- shell
    - Some usedful shell scripts

## How to run
### Dependencies
- [python3/3.6.5](https://www.python.org/downloads/release/python-365/)
- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)(currently tested on version 1.0)
- [Moses](http://www.statmt.org/moses/?n=moses.releases) (train PBSMT model)
- [cuda](https://developer.nvidia.com/cuda-92-download-archive)(currently tested on version 9.2)
- [gcc](https://gcc.gnu.org/install/)(currently tested on version 5.5.0)
- [cuda](https://developer.nvidia.com/cuda-92-download-archive) (currently tested on version 9.2)
### Data Augmentation
Run ```aug_embed.ipynb``` for word embeddings augmentation and ```aug_tfidf.ipynb``` for TF-IDF augmentation

For TF-IDF augmentation, install nlpaug:https://github.com/makcedward/nlpaug
```
pip install numpy requests nlpaug
```
or
```
pip install numpy git+https://github.com/makcedward/nlpaug.git
```

### Embedding Training
1. Download the XLM model:
```
git clone https://github.com/facebookresearch/XLM.git
```
2. Go into the XLM folder:
```
cd XLM
```
3. Install the fastBPE:
```
cd tools
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```
4. Prepare the dataset for BPE encoding:
```shell
cp ../shell/prepare_data.sh prepare_data.sh
chmod +x prepare_data.sh
./prepare_data.sh
```
5. BPE encoding for Bora, Spanish and Bora-Spanish pair:
```shell
cp ../shell/bpe.sh bpe.sh
chmod +x bpe.sh
./bpe.sh
```
6. Train the XLM model with this code:
This scripts may take 7 days to finish the running.
```shell
python train.py --exp_name XLM_bora_es --dump_path ./dumped --data_path data/processed/XLM_bora_es/60k --lgs 'bora_es' --mlm_steps 'bora_es' --max_epoch 5000
```
7. Get the word embedding:
Note: the name for the folder containing the language model is arbitrary. You can get the name from the log of the training. Let's say the folder has the name abcdef here. Remember to replace the checkpoint's path in the get_emb.py with the correct one.
```shell
cp ../shell/get_emb.sh get_emb.sh
chmod +x get_emb.sh
./get_emb.sh
```

### Translation Model 
#### Unsupervised PBSMT:
The related GitHub repositioies are [https://github.com/facebookresearch/UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT) and [https://github.com/facebookresearch/MUSE](https://github.com/facebookresearch/MUSE).

Running the PBSMT approach requires to have a working version of Moses. On some systems Moses is not very straightforward to compile, and it is sometimes much simpler to download the binaries directly.

Once you have a working version of Moses, edit the ```MOSES_PATH``` variable inside the ```PBSMT/run.sh``` script to indicate the location of Moses directory. Then, go back to the root path of the project and simply run:
```
cd PBSMT
./run.sh
```
The script will successively:

- Check Moses files
- Prepare monolingual data
    - Copy tokenized data to related path
    - Learn truecasers and apply them on monolingual data
    - Learn and binarize language models for Moses decoding
- Download and prepare parallel data (for evaluation):
    - Copy tokenized data to related path
    - Truecase parallel data
- Copy embeddings to the related path
- Run MUSE to generate cross-lingual embeddings
- Generate an unsupervised phrase-table using MUSE alignments
- Run Moses
    - Create Moses configuration file
    - Run Moses on test sentences
    - Detruecase translations
- Evaluate translations

```run.sh``` contains a few parameters defined at the beginning of the file:
- ```MOSES_PATH``` folder containing Moses installation
- ```N_MONO``` number of monolingual sentences for each language (default 185000)
- N_THREADS number of threads in data preprocessing (default 48)
- SRC source language (default Bora)
- TGT target language (default Spanish)

The result is stored in ```./PBSMT/moses_train_bora-es```.
#### Unsupervised Neural Machine Translation Model:
1. Go to the root path of the project:
2. Download the code for Unsupervised Neural Machine Translation Model:
```shell
git clone https://github.com/artetxem/undreamt.git
```
3. Train the neural matchine translation model:
This script may take 5 days to run.
```shell
cd undreamt
mkdir dumped
python train.py --embedding_size 512 --save_interval 10000 --log_interval 1000 --src ../data/bora/bora.train --trg ../data/es/es.train --src_embeddings ../XLM/emb/bora.emb.vec --trg_embeddings ../XLM/emb/es.emb.vec --save ./dumped/es_bora --cuda
```
4. Translate the test set with this model:
```shell
cp ../augmented_tokenized_data/bora.test input
python translate.py ./dumped/es_bora_aug.final.src2trg.pth --input ./input --output ./output
```
5. Download the output and compute the Bleu score from: https://www.letsmt.eu/Bleu.aspx

## Team Members
- Changhao Liang (U16843909) 
- Jun Xiao (U85900288) 
- Qi Yin (U31787103)
- Yuang Liu  (U99473611)
- Zixiang Wei (U97992068)