mkdir data
cd data
mkdir bora_es
cat ../../augmented_tokenized_data/bora.train >> bora_es/bora_es.train
cat ../../augmented_tokenized_data/es.train >> bora_es/bora_es.train
cat ../../augmented_tokenized_data/bora.valid >> bora_es/bora_es.valid
cat ../../augmented_tokenized_data/es.valid >> bora_es/bora_es.valid
cat ../../augmented_tokenized_data/bora.test >> bora_es/bora_es.test
cat ../../augmented_tokenized_data/es.test >> bora_es/bora_es.test
cd ..
