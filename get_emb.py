import os
import torch
from logging import getLogger
from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

logger = getLogger()


# NOTE: remember to replace the model path here
model_path = './dumped/XLM_bora_es/abcedf/checkpoint.pth'
reloaded = torch.load(model_path)
params = AttrDict(reloaded['params'])
print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
params.n_words = len(dico)
params.bos_index = dico.index(BOS_WORD)
params.eos_index = dico.index(EOS_WORD)
params.pad_index = dico.index(PAD_WORD)
params.unk_index = dico.index(UNK_WORD)
params.mask_index = dico.index(MASK_WORD)

# build model / reload weights
model = TransformerModel(params, dico, True, True)
model.eval()
model.load_state_dict(reloaded['model'])

codes = "./data/processed/XLM_bora_es/60k/codes"  # path to the codes of the model
fastbpe = os.path.join(os.getcwd(), 'tools/fastBPE/fast')


def to_bpe(sentences):
    # write sentences to tmp file
    with open('./tmp/sentences', 'wb') as fwrite:
        for sent in sentences:
            fwrite.write(sent.encode('utf-8') + '\n'.encode('utf-8'))

    # apply bpe to tmp file
    os.system('%s applybpe ./tmp/sentences.bpe ./tmp/sentences %s' % (fastbpe, codes))

    # load bpe-ized sentences
    sentences_bpe = []
    with open('./tmp/sentences.bpe', encoding='utf-8') as f:
        for line in f:
            sentences_bpe.append(line.rstrip())

    return sentences_bpe

languages_vocab = ['es.vocab_1', 'es.vocab_2', 'es.vocab_3', 'bora.vocab']  # four files name for vocab
embd_name = ['es.emb_1', 'es.emb_1', 'es.emb_1', 'bora.emb.vec']
for i in range(4):
    sentences = []
    with open('../augmented_tokenized_data/' + languages_vocab[i], encoding='utf-8') as file:
        for line in file:
            line = line.rstrip()
            sentences.append(line)

    # bpe-ize sentences
    sentences_copy = sentences
    sentences = to_bpe(sentences)

    # check how many tokens are OOV
    n_w = len([w for w in ' '.join(sentences).split()])
    n_oov = len([w for w in ' '.join(sentences).split() if w not in dico.word2id])
    print('Number of out-of-vocab words: %s/%s' % (n_oov, n_w))

    # add </s> sentence delimiters
    sentences = [(('</s> %s </s>' % sent.strip()).encode('utf-8').split()) for sent in sentences]

    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w.decode('utf-8')) for w in sentences[i]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent in sentences])

    langs = None

    tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
    print(tensor.size())
    output = open('./emb/' + embd_name[i], 'wb')
    output.write(str(len(sentences)).encode('utf-8') + ' '.encode('utf-8') + str(512).encode('utf-8') + '\n'.encode('utf-8'))
    for i in range(len(sentences_copy)):
        output.write(sentences_copy[i].encode('utf-8') + ' '.encode('utf-8'))
        for q in tensor[0][i].tolist():
            output.write(str(q).encode('utf-8') + ' '.encode('utf-8'))
        output.write('\n'.encode('utf-8'))
