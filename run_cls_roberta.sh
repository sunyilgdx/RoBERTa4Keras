export PYTHONPATH=../NSP-BERT/
for i in 1 2 3 4 5
do
  python ./baselines/cls_classification_roberta.py \
  --method few-shot \
  --n_th_set $i \
  --dataset_name SST-2 \
  --batch_size 2 \
  --learning_rate 2e-5 \
  --loss_function softmax \
  --model_name roberta_large
done