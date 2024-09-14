import torch
import torch.nn as nn
from torch import Tensor, tensor


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class MSE_with_alive(nn.Module):
    def __init__(self, weight=0.7):
        super(MSE_with_alive, self).__init__()
        self.weight = weight
        self.cross1 = nn.BCEWithLogitsLoss()
        self.cross2 = nn.BCEWithLogitsLoss()

    def forward(self, inputs, target, target_label, alive, pseudo, bins):
        inputs_sum = torch.mul(bins, torch.squeeze(inputs[:, :])).sum(1)
        alive_index = 1 - (torch.clamp(inputs_sum - target, min=0, max=1) * alive).reshape(len(alive),  1)
        t = torch.tensor([1] * len(alive)).cuda()
        f = torch.tensor([0] * len(alive)).cuda()
        target = target_label
        true_index = torch.where(pseudo == 2, t, f).reshape(len(alive), 1 )
        pseudo_index = torch.where(pseudo == 1, t, f).reshape(len(alive), 1)
        inputs_true = inputs * (true_index * alive_index)
        target_true = target * (true_index * alive_index)
        inputs_pseudo = inputs * (pseudo_index * alive_index)
        target_pseudo = target * (pseudo_index)
        pseudo_count = torch.count_nonzero( pseudo_index)
        true_count = torch.count_nonzero(alive_index * true_index)
        # assert target.size == inputs.size
        loss_true = 0
        loss_pseudo = 0
        if true_count != 0:
            # loss_true = torch.sum(torch.add(torch.clamp(torch.mul(alive, torch.sub(target_true, inputs_true)),
            #                                             min=0.0), torch.mul(torch.add(alive, -1),
            #                                                                 torch.sub(target_true,
            #                                                                           inputs_true))) ** 2) / true_count
            loss_true = self.cross1(inputs_true, target_true)
        if pseudo_count != 0:
            # loss_pseudo = torch.sum(
            #     torch.add(torch.clamp(torch.mul(alive, torch.sub(target_pseudo, inputs_pseudo)), min=0.0),
            #               torch.mul(torch.add(alive, -1),
            #                         torch.sub(target_pseudo, inputs_pseudo))) ** 2) / pseudo_count
            loss_true = self.cross2(inputs_pseudo, target_pseudo)
        loss = loss_true * self.weight + loss_pseudo * (1 - self.weight)
        if true_count != 0:
            print("sur_loss:", loss.item(), "real label count:", true_count.item())
            print(inputs[1,:].detach().cpu().numpy())
        return loss


class Type_loss(nn.Module):
    def __init__(self, do_sigmoid=True):
        super(Type_loss, self).__init__()
        self.celoss1=nn.CrossEntropyLoss()
        self.celoss2=nn.CrossEntropyLoss()
        self.weight=0.5

    def forward(self, inputs, target, type,target_label, alive, pseudo, bins):
        input_sur_time=torch.squeeze(inputs)
        valid_mask = torch.where((pseudo == 2) & ((input_sur_time< target )|( alive == 0)))
        pseudo_mask = torch.where(pseudo == 1)
        valid_inputs = type[valid_mask[0],:]
        valid_label= target[valid_mask[0]]
        pseudo_inputs = type[pseudo_mask[0],:]
        pseudo_label = target[pseudo_mask[0]]
        true_count=valid_mask[0].size()[0]
        pseudo_count=pseudo_mask[0].size()[0]
        loss_true = torch.Tensor([0.0]).cuda()
        loss_pseudo = torch.Tensor([0.0]).cuda()
        valid_target_one_hot=Type_loss.get_one_hot(valid_label)
        pseudo_target_one_hot=Type_loss.get_one_hot(pseudo_label)
        if true_count != 0:
            loss_true = self.celoss1(valid_inputs.to(torch.float32),valid_target_one_hot)
        if pseudo_count != 0:
            loss_pseudo = self.celoss2(pseudo_inputs.to(torch.float32),pseudo_target_one_hot)
        loss = loss_true + loss_pseudo *  self.weight
        # if true_count != 0:
            # print(valid_inputs[0].detach().cpu().numpy(), " ",
                #   valid_label[0].detach().cpu().numpy())
        return loss
    @staticmethod
    def get_one_hot(target):
        target_one_hot=torch.zeros((target.size()[0],3)).cuda()
        index0=torch.where((target>=0)&(target<300))
        index1=torch.where((target>=300)&(target<450))
        index2=torch.where(target>=450)
        target_one_hot[index0[0],0]=1
        target_one_hot[index1[0],1]=1
        target_one_hot[index2[0],2]=1
        return target_one_hot


class Fuse_Loss(nn.Module):
    def __init__(self):
        super(Fuse_Loss, self).__init__()
        self.typeloss = Type_loss()
        self.mse = MSE_with_alive2()
        # self.mseout = MSE_with_alive2()
        self.EDice = EDiceLoss()
        # self.weight = weights

    def forward(self, seg_input, seg_target, sur_inputs, sur_target,type, sur_target_label, alive, bins, pseudo
                ,weight):
        # print(sur_inputs)
        weight.cuda()
        seg_target=seg_target.cuda()
        sur_target=sur_target.cuda()
        alive=alive.cuda()
        pseudo=pseudo.cuda()
        loss_mse=self.mse(sur_inputs, sur_target, sur_target_label, alive,pseudo, bins) 
        loss_ed=self.EDice(seg_input, seg_target)
        if type!=None:
            loss_type=self.typeloss(sur_inputs, sur_target,type, sur_target_label, alive,pseudo, bins)  
            loss = loss_mse*weight[1]+loss_ed*weight[2]+loss_type*weight[0]
        else:
            loss = loss_mse*weight[1]+loss_ed*weight[2]
            loss=torch.Tensor([0]).cuda()
        # return loss
        if type!=None:
            return [loss,loss_mse,loss_ed,loss_type]
        else:
            return [loss,loss_mse,loss_ed]


def get_cuts_label(x, cuts):
    if x == -1:
        return [0] * len(cuts)
    i = 0
    label = []
    while i < len(cuts):
        if x >= cuts[i]:
            label.append(1.0)
            x = x - cuts[i]
        else:
            label.append(x / cuts[i])
            x = 0
        i += 1
    return torch.tensor(label)



class MSE_with_alive2(nn.Module):
    def __init__(self, weight=0.5):
        super(MSE_with_alive2, self).__init__()
        self.weight = weight
        self.sigmoid=nn.Sigmoid()
        self.mse1=torch.nn.MSELoss()
        self.mse2=torch.nn.MSELoss()
        self.softmax=torch.nn.Softmax(dim=1)
        


    def forward(self, inputs, target, target_label, alive, pseudo, bins):
        # print(inputs)
        # bins=torch.Tensor( [179.,  177.,  153.,  318., 1172.]).cuda().float()
        # print(bins)
        input_sur_time=inputs.reshape((inputs.size()[0]))
        # print(target.size())
        # input_sur_time=torch.squeeze(inputs)
        # print(input_sur_time)
        # input_sur_time = torch.mul(bins, torch.squeeze(self.sigmoid(inputs[:, :]))).sum(1)
        valid_mask = torch.where((pseudo == 2) & ((input_sur_time< target )|( alive == 0)))
        pseudo_mask = torch.where(pseudo == 1)
        true_count=valid_mask[0].size()[0]
        pseudo_count=pseudo_mask[0].size()[0]
        loss_true = torch.Tensor([0.0]).cuda()
        loss_pseudo = torch.Tensor([0.0]).cuda()
        if true_count != 0:
            # print("MAEK",valid_mask[0])
        # print(input_sur_time.size())
            valid_inputs = input_sur_time[valid_mask[0]]
            valid_inputs2 = inputs[valid_mask[0]]
            valid_label= target[valid_mask[0]]
            loss_true = self.mse1(valid_inputs.to(torch.float32), valid_label.to(torch.float32))
        if pseudo_count != 0:
            pseudo_label = target[pseudo_mask[0]]
            pseudo_inputs = input_sur_time[pseudo_mask[0]]
            loss_pseudo = self.mse2(pseudo_inputs.to(torch.float32), pseudo_label.to(torch.float32))
        loss = loss_true + loss_pseudo *self.weight
        if true_count != 0:
            print(valid_inputs[0].detach().cpu().numpy(), " ",
                  valid_label[0].detach().cpu().numpy())

        return loss

    @staticmethod
    def get_suf_type_label(times):
        if times >= 0 and times < 300:
            return 0
        if times >= 300 and times < 450:
            return 1
        if times >= 450:
            return 2

    def metric(self, inputs, target, type_input,target_label, alive, pseudo, bins):
        mses = []
        rights1 = []
        rights2 = []
        input_sur_time=inputs.reshape((inputs.size()[0]))
        valid_mask = torch.where((pseudo == 2) & ((input_sur_time < target) | (alive == 0)))
        if valid_mask[0].size()[0]!=0:
            valid_inputs_time = input_sur_time[valid_mask[0]]
            valid_type = type_input[valid_mask[0],:]
            valid_type=self.softmax(valid_type)
            valid_target = target[valid_mask[0]]
            for j in range(valid_target.size(0)):
                mse = torch.sub(valid_inputs_time[j], valid_target[j]).pow(2)
                a=MSE_with_alive3.get_suf_type_label(valid_inputs_time[j])
                b=MSE_with_alive3.get_suf_type_label(valid_target[j])
                if a ==b:
                    rights1.append(1)
                else:
                    rights1.append(0)
                if torch.max(valid_type[j,:],0)[1]==b:
                    rights2.append(1)
                else:
                    rights2.append(0)
                mses.append(mse)
        return mses,rights1 ,rights2
    


class BCELoss_(nn.Module):
    def __init__(self):
        super(BCELoss_, self).__init__()

    def forward(self, logits, target):
        loss = -  target * torch.log(logits) - \
               (1 - target) * torch.log(1 - logits)
        loss = loss.mean()
        return loss
    

class MSE_with_alive3(nn.Module):
    def __init__(self, weight=1):
        super(MSE_with_alive3, self).__init__()
        self.weight = weight
        self.cross1 = nn.BCEWithLogitsLoss()
        self.cross2 = nn.BCEWithLogitsLoss()
        self.sigmoid=nn.Sigmoid()

    def forward(self, inputs, target, target_label, alive, pseudo, bins):
        input_sur_time = torch.mul(bins, torch.squeeze(self.sigmoid(inputs[:, :]))).sum(1)
        valid_mask = torch.where((pseudo == 2) & ((input_sur_time < target) | (alive == 0)))
        pseudo_mask = torch.where(pseudo == 1)
        valid_inputs = inputs[valid_mask[0], :]
        valid_inputs_time = input_sur_time[valid_mask[0]]
        valid_target = target[valid_mask[0]]
        valid_label = target_label[valid_mask[0], :]
        pseudo_inputs = inputs[pseudo_mask[0], :]
        pseudo_label = target_label[pseudo_mask[0], :]
        true_count = valid_mask[0].size()[0]
        pseudo_count = pseudo_mask[0].size()[0]
        loss_true = torch.Tensor([0.0]).cuda()
        loss_pseudo = torch.Tensor([0.0]).cuda()
        if true_count != 0:
            loss_true = self.cross1(valid_inputs, valid_label)
        if pseudo_count != 0:
            loss_pseudo = self.cross2(pseudo_inputs, pseudo_label)
        loss = loss_true * self.weight + loss_pseudo * (1 - self.weight)
        if true_count != 0:
            # print("sur_loss:", loss.item(), "real label count:", true_count)
            print(valid_inputs[0, :].detach().cpu().numpy(), " ", valid_inputs_time[0].detach().cpu().numpy(), " ",
                  valid_target[0].detach().cpu().numpy())
        return loss

    @staticmethod
    def get_suf_type_label(times):
        if times >= 0 and times < 300:
            return 0
        if times >= 300 and times < 450:
            return 1
        if times >= 450:
            return 2

    def metric(self, inputs, target, target_label, alive, pseudo, bins):
        mses = []
        rights = []
        input_sur_time = inputs
        # input_sur_time = torch.mul(bins, torch.squeeze(self.sigmoid(inputs[:, :]))).sum(1)
        valid_mask = torch.where((pseudo == 2) & ((input_sur_time < target) | (alive == 0)))
        # valid_inputs = inputs[valid_mask[0], :]
        valid_inputs_time = input_sur_time[valid_mask[0]]
        valid_target = target[valid_mask[0]]
        for j in range(valid_target.size(0)):
            mse = torch.sub(valid_inputs_time[j], valid_target[j]).pow(2)
            if MSE_with_alive3.get_suf_type_label(valid_inputs_time[j]) ==\
                    MSE_with_alive3.get_suf_type_label(valid_target[j]):
                rights.append(1)
            else:
                rights.append(0)
            mses.append(mse)
        return mses,rights


class MSE_with_alive4(nn.Module):
    def __init__(self, weight=1):
        super(MSE_with_alive4, self).__init__()
        self.weight = weight
        self.mse1=torch.nn.MSELoss()
        self.mse2=torch.nn.MSELoss()

    def forward(self, inputs, target, target_label, alive, pseudo, bins):
        input_sur_time = inputs.resize((2))
        valid_mask = torch.where((pseudo == 2) & ((input_sur_time < target )|( alive == 0)))
        pseudo_mask = torch.where(pseudo == 1)
        valid_inputs = input_sur_time[valid_mask[0]]
        valid_inputs2 = inputs[valid_mask[0]]
        valid_label= target[valid_mask[0]]
        pseudo_inputs = input_sur_time[pseudo_mask[0]]
        pseudo_label = target[pseudo_mask[0]]
        true_count=valid_mask[0].size()[0]
        pseudo_count=pseudo_mask[0].size()[0]
        loss_true = torch.Tensor([0.0]).cuda()
        loss_pseudo = torch.Tensor([0.0]).cuda()
        if true_count != 0:
            loss_true = self.mse1(valid_inputs, valid_label)
        if pseudo_count != 0:
            loss_pseudo = self.mse2(pseudo_inputs, pseudo_label)
        loss = loss_true * self.weight + loss_pseudo * (1 - self.weight)
        if true_count != 0:
            # print("sur_loss:", loss.item(), "real label count:", true_count)
            print(valid_inputs[0].detach().cpu().numpy(), " ",valid_label[0].detach().cpu().numpy())

        return loss

    @staticmethod
    def get_suf_type_label(times):
        if times >= 0 and times < 300:
            return 0
        if times >= 300 and times < 450:
            return 1
        if times >= 450:
            return 2

    def metric(self, inputs, target, target_label, alive, pseudo, bins):
        mses = []
        rights = []
        input_sur_time = inputs.resize((2))
        valid_mask = torch.where((pseudo == 2) & ((input_sur_time < target) | (alive == 0)))
        # valid_inputs = inputs[valid_mask[0], :]
        valid_inputs_time = input_sur_time[valid_mask[0]]
        valid_target = target[valid_mask[0]]
        for j in range(valid_target.size(0)):
            mse = torch.sub(valid_inputs_time[j], valid_target[j]).pow(2)
            if MSE_with_alive3.get_suf_type_label(valid_inputs_time[j]) ==\
                    MSE_with_alive3.get_suf_type_label(valid_target[j]):
                rights.append(1)
            else:
                rights.append(0)
            mses.append(mse)
        return mses,rights




