import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from modelscope.msdatasets import MsDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()

    # 初始化分布式环境
    torch.distributed.init_process_group(backend='nccl')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # 加载数据集
    trn_d = MsDataset.load('DAMO_NLP/yf_dianping', subset_name='default', split='train')
    val_d = MsDataset.load('DAMO_NLP/yf_dianping', subset_name='default', split='validation')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # 定义数据集类
    class NewsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = str(self.texts[item])
            label = self.labels[item]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    # 收集文本和标签
    tr_text, val_text, tr_label, val_label = [], [], [], []
    for i in trn_d:
        tr_text.append(i['sentence'])
        tr_label.append(i['label'])
    for i in val_d:
        val_text.append(i['sentence'])
        val_label.append(i['label'])

    # 创建 dataset 字典
    dataset = {
        'train': {
            'text': tr_text,
            'label': tr_label
        },
        'validation': {
            'text': val_text,
            'label': val_label
        }
    }

    # 创建数据加载器
    max_len = 32  # 可以适当提高max_len以捕捉更多上下文
    batch_size = 4  # 减小batch_size以适应显存
    train_dataset = NewsDataset(
        texts=dataset['train']['text'],
        labels=dataset['train']['label'],
        tokenizer=tokenizer,
        max_len=max_len
    )
    val_dataset = NewsDataset(
        texts=dataset['validation']['text'],
        labels=dataset['validation']['label'],
        tokenizer=tokenizer,
        max_len=max_len
    )

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler
    )

    # 加载BERT模型
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese")
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 训练循环
    scaler = torch.cuda.amp.GradScaler()  # 使用混合精度训练
    for epoch in range(3):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            #print(input_ids.shape,attention_mask.shape,labels.shape)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if local_rank == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        # 验证
        model.eval()
        val_sampler.set_epoch(epoch)
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        if local_rank == 0:
            print(f'Validation Accuracy: {correct / total}')

if __name__ == '__main__':

    main()
