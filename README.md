### 数据集使用
- 图像分类任务使用了mini_imagenet   https://www.modelscope.cn/datasets/tany0699/mini_imagenet100
- 文本任务使用了大众点评 https://www.modelscope.cn/datasets/DAMO_NLP/yf_dianping/quickstart
### 运行命令
- 图像分类任务：HF_ENDPOINT=https://hf-mirror.com python -m torch.distributed.launch --nproc_per_node=2 train.py
- 文本任务：HF_ENDPOINT=https://hf-mirror.com python -m torch.distributed.launch --nproc_per_node=2 bert.py