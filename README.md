# 算法组训练框架


## 环境配置

### 依赖库安装
```bash
python3 -m pip install -r requirements.txt --user
```


## 运行方式

### 文本摘要
```bash
python3 python train.py --config=/home/lyc/nlp-training/configs/summary-template.yaml
```

### 文本翻译
```bash
python3 python train.py --config=/home/lyc/nlp-training/configs/translation-template.yaml
```

### 实体抽取
```bash
python3 python train.py --config=/home/lyc/nlp-training/configs/ner-template.yaml
```

### 文本分类（互斥单分类）
```bash
python3 python train.py --config=/home/lyc/nlp-training/configs/classification-template.yaml
```

### 文本标签分类（多分类）
```bash
python3 python train.py --config=/home/lyc/nlp-training/configs/tag-template.yaml
```


## 训练参数

### 外部参数
项目训练入口文件为`train.py`，其中包含参数如下

参数名 | 必选 | 类型 | 默认值 | 说明
------- | ----  | ---- | ---- | ----------------------------
config | 是 | String | None | 训练参数文件地址
local_rank | 否 | Int | -1 | 用于在GPU上进行分布式培训的local_rank值
multi_task | 否 | String | None | 多任务模式训练
debug | 否| action | False | 是否开启debug模式，若开启则不记录wandb

### 内部参数
内部参数为config文件中包含的参数，其中包含参数如下

参数名 | 从属模块 | 类型 |  说明
------- | ------- | ---- | ----------------------------
task_name | train | String | 任务名称
model_name | train | String | 使用模型名称
pretrained_tag | train | String | 使用预训练模型名称
pretrained_model | train | String | 使用已经训练好的模型路径(需包含bin文件、config文件和vocab文件)
batch_size | train | Int | 训练批次包含条数
train_epochs | train | Int | 训练轮次
freeze_encoder | train | Bool | 是否使用freeze_encoder策略
gradient_checkpointing | train | Bool | 是否使用gradient_checkpointing策略
fp16 | train | Bool | 是否使用fp16策略
encode_document | train | Bool | 是否使用encode_document策略
ner_addBilstm | train | Bool | 是否使用带有addBilstm的预训练模型(仅在实体识别任务中存在)
output_dir | train | String | 输出文件及训练参数保存路径
 |  |  | 
type | eval | String | 任务类型，nlu为序列标注任务，nlg为序列生成任务
metric | eval | String | 测试指标
batch_size | eval | Int | 测试批次包含条数
num_beams | eval | Int | beam search 数量， 1为不执行beam search策略
early_stopping | eval | Bool | 每批至少完成num_beams个句子时是否停止波束搜索
 |  |  | 
corpus | data | String | 使用数据集的名称
data_dir | data | String | 使用数据集的路径
max_seq_length | data | Int | 读取数据最大长度
max_tgt_length | data | Int | 生成数据最大长度(仅在摘要和翻译任务中)
crosslingual | data | Bool | 使用哪种类型的摘要(仅在摘要任务中)
 |  |  | 
type | optimizer | String | 使用优化器的名称
lr | optimizer | Float | learning rate
weight_decay | optimizer | Float | 学习率衰减比率
num_warmup_steps | optimizer | Float | warmup策略持续执行steps
gradient_accumulation_steps | optimizer | Float | 梯度累计steps
max_grad_norm | optimizer | Float | 梯度修剪参数
 |  |  | 
cuda_devices | system | String | 使用几块gpu训练模型
distributed | system | Bool | 是否分布式加载数据


## 支持模型

### 使用方法
在对应任务的config.yaml中设定参数：
```yaml
train:
    model_name: "对应模型名称"
    pretrained_tag: "对应支持预训练模型"
```

### nezha
华为诺亚方舟实验室开源的基于BERT的中文预训练语言模型NEZHA

##### 支持任务：
- 实体抽取
- 文本分类（互斥单分类）
- 文本标签分类（多分类）

##### 支持预训练模型：
- huawei/nezha-en-base: 英文基础模型
- huawei/nezha-zh-base: 中文基础模型
- huawei/nezha-zh-large: 中文增强模型(24层版本)

### hfl
哈工大讯飞联合实验室发布的中文训练Whole Word Masking (wwm) BERT以及ROBERTA模型

##### 支持任务：
- 实体抽取
- 文本分类（互斥单分类）
- 文本标签分类（多分类）

##### 支持预训练模型：
- hfl/chinese-bert-wwm: 使用中文维基百科数据集训练的中文bert-wwm基础模型
- hfl/chinese-bert-wwm-ext: 使用中文维基百科，其他百科、新闻、问答等数据，总词数达5.4B的EXT数据集训练的中文bert-wwm基础模型
- hfl/chinese-roberta-wwm-ext: 使用EXT数据集训练的中文萝卜塔RoBERTa-wwm基础模型
- hfl/chinese-roberta-wwm-ext-large: 使用EXT数据集训练的中文萝卜塔RoBERTa-wwm增强模型(24层版本)

### mbart
FaceBook发布的多语言BART模型

##### 支持任务：
- 文本翻译
- 文本摘要

##### 支持预训练模型：
- facebook/mbart-large-cc25: 预训练的多语言mbart模型(24层版本)

### mT5
谷歌实验室发布的多语言T5模型

##### 支持任务：
- 文本翻译
- 文本摘要

##### 支持预训练模型：
- google/mt5-base: 标准版多语言T5模型
- google/mt5-large: 增强版多语言T5模型

## PET Training模式

作为一种比较特殊的训练模式，详情可参考论文[It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners](https://arxiv.org/abs/2009.07118)，严格意义上来说谷歌的T5模型也算在这个范畴之内。

简而言之，将所有问题全部转化为完形填空(cloze)问题，充分利用一些预训练模型的MLM Head。例如对于新闻分类问题，我们将其转化为如下形式。

```python
原文：北京时间12月4日，2020至2021赛季中国女排超级联赛开始了第三阶段比赛较量，八强战比赛格外引人注目，首日比赛将有四场强强对话，在率先进行的一场比赛中，卫冕冠军天津女排与前联赛冠军浙江女排展开争夺，前两局天津女排顺风顺水拿下，第三局大比分浙江女排一直在比分上取得领先，天津女排局末阶段发力追平比分，浙江女排多次面对局点未能把握机会，最终天津女排两分险胜对手，大比分直落三局击败浙江女排，三局比分分别是25：19、25：10、32：30。
转换：下面是一则____新闻，北京时间12月4日，2020至2021赛季中国女排超级联赛开始了第三阶段比赛较量，八强战比赛格外引人注目，首日比赛将有四场强强对话，在率先进行的一场比赛中，卫冕冠军天津女排与前联赛冠军浙江女排展开争夺，前两局天津女排顺风顺水拿下，第三局大比分浙江女排一直在比分上取得领先，天津女排局末阶段发力追平比分，浙江女排多次面对局点未能把握机会，最终天津女排两分险胜对手，大比分直落三局击败浙江女排，三局比分分别是25：19、25：10、32：30。
期望补全的词：体育
```

通过这种转换方式，将原来的新闻分类问题转换为了一个完形填空问题，其形式与Mask Language Model非常相似，而大部分的预训练模型的目标任务之一就是MLM，因此充分利用这个优势即可达到小样本甚至零样本的效果。

使用PET Training模式，需要在训练数据目录下提供一个label_map文件（与ner和textclf任务类似）。另外，在训练配置yaml文件中需要增加一个pet专用配置，具体形式可以参考`pet-template.yaml`

## 评价标准

### Rouge(Recall-Oriented Understudy for Gisting Evaluation)   

#### 定义

rouge意为面向召回的要点评估理解，是一种评估模型性能的方法，该方法通常用于生成摘要或机器翻译。

#### 使用方法
在文本摘要或文本翻译任务的config.yaml中设定参数：
```yaml
eval:
    metric: "rouge"
```

#### 返回结果
返回rouge-1、rouge-2和rouge-l三个评测结果，例如：
```
{"rouge-1": ***, "rouge-2": ***, "rouge-l": ***}
```

### BLEU (Bilingual Evaluation Understudy) 

#### 定义

BLEU意思是双语评估替补。所谓Understudy (替补)，意思是代替人进行翻译结果的评估。尽管这项指标是为翻译而发明的，但它可以用于评估一组自然语言处理任务生成的文本。

#### 使用方法
在文本摘要或文本翻译任务的config.yaml中设定参数：
```yaml
eval:
    metric: "bleu"
```

#### 返回结果
返回rouge-1、rouge-2和rouge-l三个评测结果，例如：
```
{"bleu": ***}
```

### p-r-f (precision\recall\f1) 

#### 定义

准确率、召回率和f1值。

#### 使用方法
在实体抽取、文本分类(单分类)任务的config.yaml中设定参数：
```yaml
eval:
    metric: "p-r-f"
```

在文本分类(多分类)任务的config.yaml中设定参数：
```yaml
eval:
    metric: "micro"  # mode 属于 ['micro', 'macro']
```
其中
'micro'为通过计算“正确，错误和否定”的总数来全局计算指标。

'macro'为计算每个种类的p-r-f，并找到其平均值。

#### 返回结果
返回rouge-1、rouge-2和rouge-l三个评测结果，例如：
```
{"eval_precision": ***, "eval_recall": ***, "eval_f1": ***}
```
