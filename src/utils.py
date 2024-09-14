import os
import pathlib
import pprint

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path


from matplotlib import pyplot as plt
from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff
from torch import distributed as dist
from torch.cuda.amp import autocast

from dataset.batch_utils import pad_batch1_to_compatible_size
from dataset.batch_utils import determinist_collate
from loss import EDiceLoss, Fuse_Loss
from models import get_norm_layer, DataAugmenter

def save_args(args):
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    del config['save_folder']
    del config['seg_folder']
    config_file = (args.save_folder / (args.exp_name + ".yaml"))
    # with config_file.open(mode="w") as file
    os.makedirs(args.save_folder)
    with open(str(config_file),mode="w+") as file:
        yaml.dump(config, file)


def master_do(func, *args, **kwargs):
    """Help calling function only on the rank0 process id ddp"""
    try:
        rank = dist.get_rank()
        if rank == 0:
            return func(*args, **kwargs)
    except AssertionError:
        # not in DDP setting, just do as usual
        func(*args, **kwargs)


def save_checkpoint(state: dict, save_folder: pathlib.Path,modul_name="model_best.pth.tar"):
    """Save Training state."""
    best_filename = f'{str(save_folder)}/'+modul_name
    torch.save(state, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # print(val)
        if val==0:
            self.val=0
            return 
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' '.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# TODO remove dependency to args
def reload_ckpt(args, model, optimizer, scheduler,pre=False):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        if pre:
            # model_maker = getattr(models, "Unet")
            # model_pre=model_maker(
            #     4, 3,
            #     width=args.width, deep_supervision=args.deep_sup,
            #     norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)
            # model_pre = torch.nn.DataParallel(model_pre).cuda()
            # model_pre.load_state_dict(checkpoint['state_dict'])
            # pretrained_dict=model_pre.parameters()
            # model_dict=model.state_dict()
            # # 1. filter out unnecessary keys
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # # 2. overwrite entries in the existing state dict
            # model_dict.update(pretrained_dict)
            # model.load_state_dict(model_dict)
            pretrained_dict=checkpoint['state_dict']
            net2_dict=model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net2_dict}
            net2_dict.update(pretrained_dict)
            model.load_state_dict(net2_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        return checkpoint['bins']
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))


def reload_ckpt_bis(ckpt, model, optimizer=None):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        try:
            checkpoint = torch.load(ckpt)
            start_epoch = checkpoint['epoch']
            # model_resnet101.load_state_dict({k.replace('module.',''):v for k,v in torch.load()['state_dict'].items()})
            # model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{ckpt}' (epoch {start_epoch})")
            return start_epoch
        except RuntimeError:
            # TO account for checkpoint from Alex nets
            print("Loading model Alex style")
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_metrics(preds, targets, patient, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp)
            haussdorf_dist = np.nan

        else:
            preds_coords = np.argwhere(preds[i])
            targets_coords = np.argwhere(targets[i])
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

            tp = np.sum(l_and(preds[i], targets[i]))
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[HAUSSDORF] = haussdorf_dist
        metrics[DICE] = dice
        metrics[SENS] = sens
        metrics[SPEC] = spec
        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list


class WeightSWA(object):
    """
    SWA or fastSWA
    Taken from https://github.com/benathi/fastswa-semi-sup
    """

    def __init__(self, swa_model):
        self.num_params = 0
        self.swa_model = swa_model  # assume that the parameters are to be discarded at the first update

    def update(self, student_model):
        self.num_params += 1
        print("Updating SWA. Current num_params =", self.num_params)
        if self.num_params == 1:
            print("Loading State Dict")
            self.swa_model.load_state_dict(student_model.state_dict())
        else:
            inv = 1. / float(self.num_params)
            for swa_p, src_p in zip(self.swa_model.parameters(), student_model.parameters()):
                swa_p.data.add_(-inv * swa_p.data)
                swa_p.data.add_(inv * src_p.data)

    def reset(self):
        self.num_params = 0


def save_metrics(epoch, metrics, swa, writer, current_epoch, teacher=False, save_folder=None):
    metrics1 = list(zip(*metrics[0]))
    metrics2 = metrics[1]
    metrics3 = metrics[2]
    metrics4 = metrics[3]
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics1 = [torch.tensor(dice, device="cpu").numpy() for dice in metrics1]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics1 = {key: value for key, value in zip(labels, metrics1)}
    # print(metrics)
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics1.values(), labels=metrics1.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics1.items()])
    metrics2 =np.array([torch.tensor(item, device="cpu").numpy() for item in metrics2]).mean()
    metrics3 =np.array( [torch.tensor(item, device="cpu").numpy() for item in metrics3]).mean()
    metrics4 =np.array( [torch.tensor(item, device="cpu").numpy() for item in metrics4]).mean()
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
              [f"{key} : {np.nanmean(value)}" for key, value in metrics1.items()],f"mse: {metrics2}",f"acc1: {metrics3}",f"acc2: {metrics4}" ,file=f)
    for key, value in metrics1.items():
        tag = f"val{'_teacher' if teacher else ''}{'_swa' if swa else ''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)
    print(f"Epoch {current_epoch} :mse {metrics2}")
    print(f"Epoch {current_epoch} :acc1 {metrics3}")
    print(f"Epoch {current_epoch} :acc2 {metrics3}")
    return [metrics1,metrics2,metrics3,metrics4]


def generate_segmentations(data_loader, model, writer, args):
    metrics_list = []
    for i, batch in enumerate(data_loader):
        # measure data loading time
        inputs = batch["image"]
        patient_id = batch["patient_id"][0]
        ref_path = batch["seg_path"][0]
        crops_idx = batch["crop_indexes"]
        inputs, pads = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.cuda()
        with autocast():
            with torch.no_grad():
                if model.deep_supervision:
                    pre_segs, _ = model(inputs)
                else:
                    pre_segs = model(inputs)
                pre_segs = torch.sigmoid(pre_segs)
        # remove pads
        maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
        segs = torch.zeros((1, 3, 155, 240, 240))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
        segs = segs[0].numpy() > 0.5

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)
        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)
        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])
        patient_metric_list = calculate_metrics(segs, refmap, patient_id)
        metrics_list.append(patient_metric_list)
        labelmap.CopyInformation(ref_seg_img)
        print(f"Writing {args.seg_folder}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{args.seg_folder}/{patient_id}.nii.gz")
    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    writer.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    writer.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict()
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer.add_scalar(f"benchmark_{metric}/{label}", score)
    df.to_csv((args.save_folder / 'results.csv'), index=False)


def update_teacher_parameters(model, teacher_model, global_step, alpha=0.99 / 0.999):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    # print("teacher updated!")


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]
