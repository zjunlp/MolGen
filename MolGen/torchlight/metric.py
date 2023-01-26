# from abc import ABC, ABCMeta, abstractclassmethod
import torch
import numpy as np
from abc import ABC, abstractmethod, ABCMeta

class Metric(metaclass=ABCMeta):
    """
    -   reset() in the begining of every epoch.
    -   update_per_batch() after every batch.
    -   update_per_epoch() after every epoch.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update_per_batch(self, output):
        pass

    @abstractmethod
    def update_per_epoch(self):
        pass

class Top_K_Metric(Metric):
    """
    Stores accuracy (score), loss and timing info
    """
    def __init__(self, topnum=[1,3,10]):
        super().__init__()
        # assert len(topnum) == 3
        self.topnum = topnum
        self.k_num = len(self.topnum)
        self.reset()

    def reset(self):
        self.total_loss = 0
        self.correct_list = [0] * self.k_num
        self.acc_list = [0] * self.k_num
        self.acc_all = 0
        self.num_examples = 0
        self.num_epoch = 0

        self.mrr = 0
        self.mr = 0
        self.mrr_all = 0
        self.mr_all = 0

    def update_per_batch(self, loss, ans, pred):
        self.total_loss += loss
        self.num_epoch += 1
        self.top_k_list = self.batch_accuracy(pred, ans)
        self.num_examples += self.top_k_list[0].shape[0]
        for i in range(self.k_num):
            self.correct_list[i] += self.top_k_list[i].sum().item()

        # mrr
        mrr_tmp, mr_tmp =  self.batch_mr_mrr(pred, ans)
        self.mrr_all += mrr_tmp.sum().item()
        self.mr_all += mr_tmp.sum().item()



    def update_per_epoch(self):
        for i in range(self.k_num):
            self.acc_list[i] = 100 * (self.correct_list[i] / self.num_examples)

        self.mr = self.mr_all / self.num_examples
        self.mrr = self.mrr_all / self.num_examples
        self.total_loss = self.total_loss / self.num_epoch
        self.acc_all = sum(self.acc_list)


    def batch_accuracy(self, predicted, true):
        """ Compute the accuracies for a batch of predictions and answers """
        if len(true.shape) == 3:
            true = true[0]
        _, ok = predicted.topk(max(self.topnum), dim=1)
        agreeing_all = torch.zeros([predicted.shape[0], 1], dtype=torch.float).cuda()
        top_k_list = [0]*self.topnum
        for i in range(max(self.topnum)):
            tmp = ok[:, i].reshape(-1, 1)
            agreeing_all += true.gather(dim=1, index=tmp)
            for k in range(self.k_num):
                if i == self.topnum[k] - 1:
                    top_k_list[k] = (agreeing_all * 0.3).clamp(max=1)
                    break

        return top_k_list



    def batch_mr_mrr(self, predicted, true):
        if len(true.shape) == 3:
            true = true[0]

        # 计算
        top_rank = predicted.shape[1]
        batch_size = predicted.shape[0]
        _, predict_ans_rank = predicted.topk(top_rank, dim=1) # 答案排名的坐标 batchsize * 500
        _, real_ans = true.topk(1, dim=1) # 真正的答案：batchsize * 1

        # 扩充维度
        real_ans = real_ans.expand(batch_size, top_rank)
        ans_different = torch.abs(predict_ans_rank - real_ans)
        # 此时为0的位置就是预测正确的位置
        _, real_ans_list = ans_different.topk(top_rank, dim=1) #此时最后一位的数值就是正确答案在预测答案里面的位置,为 0
        real_ans_list = real_ans_list + 1.0
        mr = real_ans_list[:,-1].reshape(-1,1).to(torch.float64)
        mrr = 1.0 / mr
        # pdb.set_trace()

        return mrr,mr


if __name__ == '__main__':
    pass