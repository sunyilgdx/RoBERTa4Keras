# RoBERTa4Keras
使用了苏神的[bert4keras](https://github.com/bojone/bert4keras)框架
分别借鉴了并改动了
- [keras_roberta](https://github.com/midori1/keras_roberta)的转换方法 (由fairseq版的RoBERETa pytorch转为原版BERT Tensorflow)
- [RobertaTokenizer](https://github.com/clearwho/RobertaTokenizer)的分词方法
具体细节后续慢慢更新

## 核心思路
- 使用英文RoBERTa、GPT2、BART等模型的bpe分词
- 传入custom position id, 将position id设置为[2, max_length]
- 抛弃segment embeddings, 将segment_vocab_size=0
- 将**fairseq**的pytorch RoBERTa转为google原版的BERT

## 使用例子
简单写了一个文本分类的代码，可以直接运行[cls_classification_roberta.py](https://github.com/sunyilgdx/RoBERTa4Keras/blob/main/cls_classification_roberta.py)，也可以使用[run_cls_roberta.sh](https://github.com/sunyilgdx/RoBERTa4Keras/blob/main/run_cls_roberta.sh)(可能需要修改相对路径)
