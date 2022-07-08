# Lab5: fine-tune model automatically

## Prepare model
git-lfs clone https://huggingface.co/bert-base-cased

# 若从/huggingface.co中拉下来的model 需要删除 .git, .gitattributes，否则会报错：
# If this is a private repository, make sure to pass a token having permission to this repo with `use_auth_token` or log in with `huggingface-cli login` and pass `use_auth_token=True`.
cd bert-base-cased
rm -rf .git
rm -rf .gitattributes

### 压缩成tar.gz

tar -czvf bert-base-cased.tar.gz bert-base-cased

## Update constant.py parameter

MODEL_NAME_DIR = "bert-base-cased" #更新model文件夹名字
INPUT_DATA_FILENAME_PY = "squad.py" # 因为数据集相同，故不需修改s

## start to training

python transfer_learning_qa.py --model-dir "./bert_model_dir" --train "/pytorch/fine-tunable/01-fine-tunable/script/sourcedir/datasets/squad/" --pretrained-model "/pytorch/fine-tunable/model-targz" --epochs 1 --batch-size 16
