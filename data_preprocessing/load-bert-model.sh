mkdir -p "./data/BERT"
curl 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip' -o './data/BERT/uncased_L-12_H-768_A-12.zip';
unzip "./data/BERT/uncased_L-12_H-768_A-12.zip" -d "./data/BERT";
rm "./data/BERT/uncased_L-12_H-768_A-12.zip";