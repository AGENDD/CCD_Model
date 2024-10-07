import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from Models.GCN_self import GCN_self
from Models.MF_self import MF_self
from Models.Transformer_self import Transformer_self
from Models.VAE_self import VAE_self

class UserItemDataset(Dataset):
    def __init__(self, csv_file, num_users, num_items, num_negatives=1):
        self.data = pd.read_csv(csv_file)
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        self.neg_items = {}
        
        for i in range(num_users):
            self.neg_items[i] = []
            
            for j in range(num_items):
                if(i,j) not in set(zip(self.data['user'], self.data['item'])):
                    self.neg_items[i].append(j)

            # print(f"user{i}:neg{len(self.neg_items[i])}")
        
        
        print(self.data)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user = self.data.iloc[idx, 0]
        pos_item = self.data.iloc[idx, 1]
        
        for _ in range(self.num_negatives):
            neg_item = random.choice(self.neg_items[user])

        return {'user': torch.tensor(user, dtype=torch.long), 
                'pos_item': torch.tensor(pos_item, dtype=torch.long), 
                'neg_item': torch.tensor(neg_item, dtype=torch.long)}

# 加载数据集
csv_file = 'Data/user_item_pairs.csv'
num_users = 100
num_items = 1000
num_negatives = 1

print(f"loading dataset")
dataset = UserItemDataset(csv_file, num_users, num_items, num_negatives)
dataloader = DataLoader(dataset, batch_size=32,shuffle=True)

# 示例参数
embedding_dim = 8
# 初始化模型和优化器

model_id = 1

if(model_id == 0):
    model = MF_self(num_users, num_items, embedding_dim)
elif(model_id == 1):
    model = GCN_self(num_users, num_items, embedding_dim)
elif(model_id == 2):
    model = Transformer_self(num_users, num_items, embedding_dim, nhead=4, num_layers=2)
elif(model_id == 3):
    model = VAE_self(num_users, num_items, embedding_dim)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"training")
# 训练模型
model.train()
for epoch in range(100):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        pos_score, neg_score = model(batch)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

print("Training complete.")
