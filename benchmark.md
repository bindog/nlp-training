# Benchmark


### 文本摘要
corpus | model_name | BLEU | Rouge-1 | Rouge-2 | Rouge-L
---- | --------  | ---- | ---- | ---- | ----
xx | xxx | xx | xx | xx| xx

### 文本翻译
corpus | model_name | BLEU | Rouge-1 | Rouge-2 | Rouge-L
---- | --------  | ---- | ---- | ---- | ----
TED2013 | facebook/mbart-large-cc25 | 41.4 | - | - | -

### 实体抽取
corpus | model_name | precision | recall | f1-score
---- | --------  | ---- | ---- | ----
ner | huawei/nezha-en-base | 72.56 | 65.13 | 68.65
ner | hfl/chinese-roberta-wwm-ext | 65.97 | 64.3 | 65.13


### 文本分类（互斥单分类）
corpus | model_name | precision | recall | f1-score
---- | --------  | ---- | ---- | ----
tag_zh | huawei/nezha-zh-base | 77.31 | 77.31 | 77.31
tag_zh | simple(lr=0.1, bs=64) |  87.02 | 85.50 | 85.90

### 文本标签分类（多分类）
corpus | model_name | precision | recall | f1-score
---- | --------  | ---- | ---- | ----
RTMACP | huawei/nezha-en-base | 77.81 | 57.09 | 65.86
RTMACP | hfl/chinese-roberta-wwm-ext | 71.68 | 43.94 | 54.48
