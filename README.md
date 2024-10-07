# CCD_Model


Model inputs:
        '''
            mini_batch = {
                'user': torch.tensor([0, 1, 2]),  # 用户索引
                'pos_item': torch.tensor([1, 2, 3]),  # 正样本物品索引
                'neg_item': torch.tensor([3, 1, 0])   # 负样本物品索引
            }
        '''

Model outputs:
    pos_score, neg_score