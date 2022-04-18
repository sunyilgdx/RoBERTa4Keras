# RoBERTa4Keras
使用了苏神的[bert4keras](https://github.com/bojone/bert4keras)框架

分别借鉴了并改动了
- [keras_roberta](https://github.com/midori1/keras_roberta)的转换方法 (由fairseq版的RoBERETa pytorch转为原版BERT Tensorflow)
- [RobertaTokenizer](https://github.com/clearwho/RobertaTokenizer)的分词方法


## 核心思路
- 使用英文RoBERTa、GPT2、BART等模型的bpe分词
- 传入custom position id, 将position id设置为 \[2, max_length\]
- 抛弃segment embeddings, 将segment_vocab_size=0
- 将**fairseq**的pytorch RoBERTa转为google原版的BERT

## 使用例子
简单写了一个文本分类的代码，可以直接运行[cls_classification_roberta.py](https://github.com/sunyilgdx/RoBERTa4Keras/blob/main/cls_classification_roberta.py)，也可以使用[run_cls_roberta.sh](https://github.com/sunyilgdx/RoBERTa4Keras/blob/main/run_cls_roberta.sh)(可能需要修改相对路径)

1. 将fairseq原版的pytorch RoBERTa转为tensorflow格式
  - 从[fairseq/RoBERTa](https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.md)下载，[base](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)或[large](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)版的RoBERTa，保存至[RoBERTa4Keras/models](https://github.com/sunyilgdx/RoBERTa4Keras/tree/main/models)下.
  - 使用[convert_fairseq_roberta_to_tf.py](https://github.com/sunyilgdx/RoBERTa4Keras/blob/main/convert_fairseq_roberta_to_tf.py)将其转换为tensorflow格式，同样保存在[RoBERTa4Keras/models]()下
  - 此时，`models/roberta_large_fairseq_tf`下有四类文件，分别是`bert_config.json` `merges.txt` `vocab.json`和`roberta_large.ckpt.data/index/meta`三个文件
2. 基于[CLS]位置的fine-tuning关键步骤
  - 加载bpe分词器
  
    ```
    merges_file = r'./models/roberta_large_fairseq_tf/merges.txt'
    vocab_file = r'./models/roberta_large_fairseq_tf/vocab.json'
    tokenizer = RobertaTokenizer(vocab_file=vocab_file, merges_file=merges_file)
    ```
  - 使用bert4keras的框架中的BERT模型加载RoBERTa模型
  这里需要设置两个关键参数，`custom_position_ids=True`传入自定义position_ids， `segment_vocab_size=0`将segment embeddings维度设置为0
 
    ```
    bert = build_transformer_model(
          config_path,
          checkpoint_path,
          model=model,
          with_pool=False,
          return_keras_model=False,
          custom_position_ids=True,
          segment_vocab_size=0  # RoBERTa don't have the segment embeddings (token type embeddings).
      )
    ```
  - 分词和token编码
    这里会调用`bpe_tokenization.py`中的分词模型，将会把输入的文本编码为`<s> X </s>`这种形式
    
    ```
    token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
    ```
  - 位置编码
    这里需要把位置编码设置为\[2,max_len\]，具体原因需要到fairseq的仓库下查issues，padding使用的是1这个position id(可能也没有影响)
    
    ```
    custom_position_ids = [2 + i for i in range(len(token_ids))]
    batch_custom_position_ids = sequence_padding(batch_custom_position_ids, value=1)
    ```
  其他地方与中文BERT基本一致，只需要改一些inputs就可以了，不再赘述
