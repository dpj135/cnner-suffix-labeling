import numpy
import torch


def surrounding_spans(start:int , end:int  , distance: int, length: int):
    """Spans with given `distance` to the given `span`.
    """
    if distance == 0 :
        return

    for x_dis in range(0,distance+1):
        y_dis = distance - x_dis
        for x_offset in {x_dis,-x_dis}:
            for y_offset in {y_dis,-y_dis}:
                x, y = start + x_offset, end + y_offset
                if 0 <= x <= y < length:
                    yield (x, y)
def soft_crossEntropyLoss(label_score_matrix:torch.FloatTensor , predict_score_matrix:torch.FloatTensor):
        loss = torch.sum(-label_score_matrix * torch.nn.functional.log_softmax(predict_score_matrix, -1))
        return loss
class SL:
    def __init__(self,label_matrix:numpy.array,length:int,type_num:int):
        self.label_score_matrix = None
        self.label_matrix=label_matrix
        self.length=length
        self.type_num=type_num
        self.eps = 0.15#后缀标注系数
        self.L=1#平滑大小

    def one_hot(self):
        one_hot_codes = numpy.eye(self.type_num)
        self.label_score_matrix=one_hot_codes[self.label_matrix]
    def norm(self):
        # 调整socre总和为1，考虑先调用numpy.sum统一求和，可能可以提升速度
        for j in range(self.length):
            for i in range(j + 1):
                score_sum=numpy.sum(self.label_score_matrix[i][j])
                if score_sum>1:
                    self.label_score_matrix[i][j][0] -= score_sum-1
    def suffix_labeling(self):
        self.one_hot()
        for j in range(self.length):
            suffix_j=numpy.zeros(self.type_num)
            for i in range(j+1):
                if self.label_matrix[i][j]>=1:
                    suffix_j+=self.eps*self.label_score_matrix[i][j]
                else:
                    self.label_score_matrix[i][j] += suffix_j
        self.norm()
        return self.label_score_matrix
    def label_smoothing(self):
        ori_score_matrix=self.label_score_matrix.copy()
        for j in range(self.length):
            for i in range(j + 1):
                if self.label_score_matrix[i][j][0] < 1.0:
                    for cur_distance in range(1,self.L+1):
                        surrounding_spans_list=list(surrounding_spans(i,j,cur_distance,self.length))
                        for l , r in surrounding_spans_list:
                            self.label_score_matrix[l][r]+= self.eps/(self.L*len(surrounding_spans_list))*ori_score_matrix[i][j]
                        self.label_score_matrix[i][j]-=self.eps*ori_score_matrix[i][j]
        self.norm()
        return self.label_score_matrix
    def sl_and_ls(self):
        self.suffix_labeling()
        self.label_smoothing()
        return self.label_score_matrix

