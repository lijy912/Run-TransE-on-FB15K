# 加载数据集并生成实体和关系的标签编码
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class TransE(nn.Module):
    def __init__(self, num_entity, num_relation, emb_dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.entity_emb = nn.Embedding(num_entity, emb_dim)
        self.relation_emb = nn.Embedding(num_relation, emb_dim)
        self.margin = margin

    def forward(self, pos_samples, neg_samples):
        # 解析正样本数据
        pos_head = pos_samples[:, 0]
        pos_rel = pos_samples[:, 1]
        pos_tail = pos_samples[:, 2]

        # 解析负样本数据
        neg_head = neg_samples[:, 0]
        neg_rel = neg_samples[:, 1]
        neg_tail = neg_samples[:, 2]

        # 计算实体和关系的嵌入表示
        pos_head_emb = self.entity_emb(torch.tensor(pos_head)).float()
        pos_tail_emb = self.entity_emb(torch.tensor(pos_tail)).float()
        neg_head_emb = self.entity_emb(torch.tensor(neg_head)).float()
        neg_tail_emb = self.entity_emb(torch.tensor(neg_tail)).float()
        rel_emb = self.relation_emb(torch.tensor(pos_rel)).float()

        # 计算实体和关系的 L1 距离
        pos_score = torch.norm(pos_head_emb + rel_emb - pos_tail_emb, p=1, dim=1)
        neg_score = torch.norm(neg_head_emb + rel_emb - neg_tail_emb, p=1, dim=1)

        return pos_score, neg_score

    def train_step(self, pos_samples, neg_samples):
        self.optimizer.zero_grad()
        pos_score, neg_score = self.forward(pos_samples, neg_samples)
        loss = torch.sum(torch.relu(pos_score + self.margin - neg_score))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, triples, num_epoch=10, batch_size=128, lr=0.001):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # 生成实体和关系标签编码
        entity2id = {}
        relation2id = {}
        for h, r, t in triples:
            if h not in entity2id:
                entity2id[h] = len(entity2id)
            if t not in entity2id:
                entity2id[t] = len(entity2id)
            if r not in relation2id:
                relation2id[r] = len(relation2id)

        num_entity = len(entity2id)
        num_relation = len(relation2id)
        print(f"Number of entities: {num_entity}")
        print(f"Number of relations: {num_relation}")

        # 将三元组数据转换为标签编码
        triples = np.array([(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples])

        for epoch in range(num_epoch):
            np.random.shuffle(triples)
            num_batch = len(triples) // batch_size

            for i in range(num_batch):
                batch_data = triples[i * batch_size: (i + 1) * batch_size]

                # 生成负样本
                neg_samples = self.get_neg_samples(batch_data)

                # 训练一批数据
                pos_samples = torch.tensor(batch_data).long()
                neg_samples = torch.tensor(neg_samples).long()
                loss = self.train_step(pos_samples, neg_samples)

                print(f"Epoch {epoch}/{num_epoch}, Batch {i}/{num_batch}, Loss: {loss}")

    def get_neg_samples(self, batch_data, num_sample=1):
        neg_samples = []
        for head, rel, tail in batch_data:
            for i in range(num_sample):
                # 生成随机负例
                if np.random.randint(2) == 0:
                    # 更换头实体
                    while True:
                        fake_head = np.random.randint(self.entity_emb.num_embeddings)
                        if fake_head != head:
                            break
                    neg_samples.append([fake_head, rel, tail])
                else:
                    # 更换尾实体
                    while True:
                        fake_tail = np.random.randint(self.entity_emb.num_embeddings)
                        if fake_tail != tail:
                            break
                    neg_samples.append([head, rel, fake_tail])
        return neg_samples

# 加载fb15k数据集
triples = pd.read_csv("./fb15k/freebase_mtr100_mte100-train.txt", sep="\t", header=None)
print(f"Number of triples: {len(triples)}")
triples = triples.values.tolist()