cp ../get_emb.py get_emb.py
mkdir tmp
mkdir emb
head -30000 ../A-Bora-Dataset-for-Machine-Translation/augmented_tokenized_data/es.vocab > ../A-Bora-Dataset-for-Machine-Translation/augmented_tokenized_data/es.vocab_1
head -60000 ../A-Bora-Dataset-for-Machine-Translation/augmented_tokenized_data/es.vocab | tail -30000 > ../A-Bora-Dataset-for-Machine-Translation/augmented_tokenized_data/es.vocab_2
tail -13888 ../A-Bora-Dataset-for-Machine-Translation/augmented_tokenized_data/es.vocab > ../A-Bora-Dataset-for-Machine-Translation/augmented_tokenized_data/es.vocab_3
python get_emb.py
touch emb/es.emb.vec
cat es.emb_1 >> es.emb.vec
cat es.emb_2 >> es.emb.vec
cat es.emb_3 >> es.emb.vec