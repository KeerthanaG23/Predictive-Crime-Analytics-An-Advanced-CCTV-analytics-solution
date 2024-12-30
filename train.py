import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch import nn
import option
args=option.parse_args()
from tqdm import tqdm

def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss

def smooth(arr, lamda1):
    arr1 = arr[:, :-1, :]
    arr2 = arr[:, 1:, :]
    loss = torch.sum((arr2 - arr1) ** 2)
    return lamda1 * loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=200.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(-torch.abs(x))
        return torch.abs(torch.mean(-x * target + torch.clamp(x, min=0) + torch.log(tmp)))

class mgfn_loss(torch.nn.Module):
    def __init__(self, alpha):
        super(mgfn_loss, self).__init__()
        self.alpha = alpha
        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss()
        self.contrastive = ContrastiveLoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude):
        label = torch.cat((nlabel, alabel), 0)
        score = torch.cat((score_normal, score_abnormal), 0).squeeze()
        label = label.cuda()
        separate = len(abn_feamagnitude) / 2

        loss_cls = self.criterion(score, label)
        loss_con = self.contrastive(torch.norm(abn_feamagnitude, p=1, dim=2), 
                                    torch.norm(nor_feamagnitude, p=1, dim=2), 1)  # Separate normal and abnormal
        loss_con_n = self.contrastive(torch.norm(nor_feamagnitude[int(separate):], p=1, dim=2),
                                      torch.norm(nor_feamagnitude[:int(separate)], p=1, dim=2), 0)  # Cluster the same class
        loss_con_a = self.contrastive(torch.norm(abn_feamagnitude[int(separate):], p=1, dim=2),
                                      torch.norm(abn_feamagnitude[:int(separate)], p=1, dim=2), 0)

        loss_total = loss_cls + 0.001 * (0.001 * loss_con + loss_con_a + loss_con_n)
        return loss_total

def train(nloader, aloader, model, batch_size, optimizer, device, iterator=0):
    model.train()
    loss_criterion = mgfn_loss(0.0001)  # Instantiate mgfn_loss once

    total_cost = 0.0  # Initialize total cost
    total_loss_smooth = 0.0  # Initialize total smooth loss
    total_loss_sparse = 0.0  # Initialize total sparse loss
    step=0

    for step, ((ninput, nlabel), (ainput, alabel)) in tqdm(enumerate(zip(nloader, aloader))):
        input = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = model(input)  # b*32 x 2048
        
        loss_sparse = sparsity(scores[:batch_size, :, :].view(-1), batch_size, 8e-3)
        
        loss_smooth = smooth(scores, 8e-4)

        scores = scores.view(batch_size * 32 * 2, -1).squeeze()

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude) + loss_smooth + loss_sparse

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        total_cost += cost.item()
        total_loss_smooth += loss_smooth.item()
        total_loss_sparse += loss_sparse.item()

        iterator += 1

    # Return the average costs
    return total_cost / (step + 1), total_loss_smooth / (step + 1), total_loss_sparse / (step + 1)