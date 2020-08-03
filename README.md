# nlp-playground

Modified from [NEZHA](https://github.com/huawei-noah/Pretrained-Language-Model)

## environment

```bash
# install python-dev first
# sudo apt-get install python3.6-dev
# OR sudo yum -y install python36-devel
export CUDA_HOME="/usr/local/cuda-10.1"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 4a8c4ac  # some function only in old version apex
python3 -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user
```

## run

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python3 main_nezha.py \
  --task_name=ner \
  --do_train \
  --do_test \
  --data_dir=/your/bio/format/data/path \
  --bert_model=/nezha/pretrained/model/path \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --learning_rate=3e-5  \
  --num_train_epochs=10.0 \
  --output_dir=/your/model/output/path
  # --trained_model_dir=/restore/trained/model/path \
```
