mkdir data
cd data
mkdir bora
mkdir es
mkdir bora_es
cp ../../augmented_tokenized_data/bora.train ../../augmented_tokenized_data/bora.test ../../augmented_tokenized_data/bora.valid bora
cp ../../augmented_tokenized_data/es.train ../../augmented_tokenized_data/es.test ../../augmented_tokenized_data/es.valid es
cat bora/bora.train >> bora_es/bora_es.train
cat es/es.train >> bora_es/bora_es.train
cat bora/bora.valid >> bora_es/bora_es.valid
cat es/es.valid >> bora_es/bora_es.valid
cat bora/bora.test >> bora_es/bora_es.test
cat es/es.test >> bora_es/bora_es.test
cd ..
