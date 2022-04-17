export PYTHONPATH=../RoBERTa4Keras/
for i in 1 2 3 4 5
do
  python cls_classification_roberta.py \
  --dataset_name SST-2 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --model_name roberta_large
done