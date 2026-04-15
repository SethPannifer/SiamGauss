import torch
import torch.nn as nn

class QuadrupletLossbatch(nn.Module):
    def __init__(self):
        super(QuadrupletLossbatch, self).__init__()

    def forward(self, positive, neutral, negative):
        PMN_counter = [0.1,0.1,0.1]
        pos_loss = 0
        neu_loss = 0
        neg_loss = 0
        correct_loss = 0
        
        if positive is not None:
            for i in range (0,len(positive)):
                pos_loss += (positive[i] - 1) ** 2
                PMN_counter[0] = PMN_counter[0] + 1
        if neutral is not None:
            for i in range (0,len(neutral)):
                neu_loss += (neutral[i]) ** 2
                PMN_counter[1] = PMN_counter[1] + 1
        if negative is not None:
            for i in range (0,len(negative)):
                neg_loss += (negative[i] + 1) ** 2
                PMN_counter[2] = PMN_counter[2] + 1
        
        loss = (pos_loss/PMN_counter[0])+((neu_loss/PMN_counter[1]))+(neg_loss/PMN_counter[2])

        return loss

        # losses = []
        # if positive is not None:
        #     for i in range (0,len(positive)):
        #         losses.append((positive[i] - 1) ** 2)
        # if neutral is not None:
        #     for i in range (0,len(neutral)):
        #         losses.append((neutral[i]) ** 2)
        # if negative is not None:
        #     for i in range (0,len(negative)):
        #         losses.append((negative[i] + 1) ** 2)

        # loss = torch.mean(torch.cat(losses))
        # return loss

        # pos_loss = ((positive - 1)**2).sum() if positive is not None else 0
        # neu_loss = (neutral**2).sum() if neutral is not None else 0
        # neg_loss = ((negative + 1)**2).sum() if negative is not None else 0

        # PMN_counter = [
        #     positive.size(0) if positive is not None else 1,
        #     neutral.size(0) if neutral is not None else 1,
        #     negative.size(0) if negative is not None else 1
        # ]
        # loss = (pos_loss / PMN_counter[0]) + (neu_loss / (PMN_counter[1])) + (neg_loss / PMN_counter[2])
        # return loss



class QuadrupletLoss(nn.Module):
    def __init__(self):
        super(QuadrupletLoss, self).__init__()

    def forward(self, positive, neutral, negative):
        pos_loss = (positive - 1) ** 2
        neu_loss = (neutral)**2
        neg_loss = (negative + 1) ** 2
        loss = pos_loss + neu_loss + neg_loss
        return loss

