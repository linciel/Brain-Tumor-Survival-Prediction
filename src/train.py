import argparse
import os
import pathlib
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from ranger import Ranger
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import models
from config import BRATS_SUR_FOLDER
from confidence import Confidence
from dataset import get_datasets
from dataset.batch_utils import determinist_collate
from loss import EDiceLoss, Fuse_Loss
from models import get_norm_layer, DataAugmenter
from utils import save_args, AverageMeter, ProgressMeter, reload_ckpt, save_checkpoint, reload_ckpt_bis, \
    count_parameters, WeightSWA, save_metrics, generate_segmentations

parser = argparse.ArgumentParser(description='Brats Training')
# 模型的名称
parser.add_argument('-a', '--arch', metavar='ARCH', default='Unet2',
                    help='model architecture (default: Unet)')
parser.add_argument('-vt', '--val_test',  default=False,
                    help='val_test')
parser.add_argument('-test', '--test',  default=False,
                    help='test')
# Unet的基准特征宽度
parser.add_argument('--width', default=48, help='base number of features for Unet (x2 per downsampling)', type=int)
# DO not use data_aug argument this argument!!
# 加载数据时使用的线程数
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
# restart中的开始epoch数
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# 跑的总数
parser.add_argument('--epochs', default=520, type=int, metavar='N',
                    help='number of total epochs to run')
# 批处理大小
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1)')
# 初始学习率
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
# 权值衰减减少过拟合
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
# Warning: untested option!!
# 从ckp点恢复训练，参数为model_best.pth.tar的位置
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint. Warning: untested option')

# 使用的设备
parser.add_argument('--devices', required=False, type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--debug', action="store_true")
parser.add_argument('--deep_sup', action="store_true")
parser.add_argument('--no_fp16', default=False, action="store_true")
parser.add_argument('--seed', default=20, help="seed for train/val split")
parser.add_argument('--warm', default=0, type=int, help="number of warming up epochs")

parser.add_argument('--val', default=3, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=4, type=int, help="Split number (0 to 4)")
parser.add_argument('--norm_layer', default='group')
parser.add_argument('--swa', action="store_true", default=False,
                    help="perform stochastic weight averaging at the end of the training")
parser.add_argument('--swa_repeat', type=int, default=5, help="how many warm restarts to perform")
# 优化器类型
parser.add_argument('--optim', choices=['adam', 'sgd', 'ranger', 'adamw'], default='adam')
# 训练备注
parser.add_argument('--com', help="add a comment to this run!")
# 是否使用dropout
parser.add_argument('--dropout', type=float, help="amount of dropout to use", default=0.)
# 对于前30次训练是否使用worm restarts
parser.add_argument('--warm_restart', action='store_true', help='use scheduler warm restarts with period of 30')
# 是否使用全部的训练数据去训练模型
parser.add_argument('--full', default=False, action='store_true', help='Fit the network on the full training set')


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def main(args):
    """ The main training function.

    Only works for single node (be it single or multi-GPU)

    Parameters
    ----------
    args :
        Parsed arguments
    """
    # setup 读取gpu 数量

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # 设置随机数种子
    setup_seed(20)
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        # raise RuntimeWarning("This will not be able to run on CPU only")
        pass
    print(f"Working with {ngpus} GPUs")
    if args.optim.lower() == "ranger":
        # No warm up if ranger optimizer
        args.warm = 0

    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    args.exp_name = f"{'debug_' if args.debug else ''}{current_experiment_time}_" \
                    f"_fold{args.fold if not args.full else 'FULL'}" \
                    f"_{args.arch}_{args.width}" \
                    f"_batch{args.batch_size}" \
                    f"_optim{args.optim}" \
                    f"_{args.optim}" \
                    f"_lr{args.lr}-wd{args.weight_decay}_epochs{args.epochs}_deepsup{args.deep_sup}" \
                    f"_{'fp16' if not args.no_fp16 else 'fp32'}" \
                    f"_warm{args.warm}_" \
                    f"_norm{args.norm_layer}{'_swa' + str(args.swa_repeat) if args.swa else ''}" \
                    f"_dropout{args.dropout}" \
                    f"_warm_restart{args.warm_restart}" \
                    f"{'_' + args.com.replace(' ', '_') if args.com else ''}"\
                    f"_epoch{args.warm_restart}" \
                    f"{'_' + args.com.replace(' ', '_') if args.com else ''}"
    args.exp_name2 = f"{current_experiment_time}"
    
    args.save_folder = pathlib.Path(f"./runs/{args.exp_name2}")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.save_folder / "segs"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    args.save_folder = args.save_folder.resolve()
    # save_args(args)
    t_writer = SummaryWriter(str(args.save_folder))
    with open(args.save_folder / "args",'a') as f:
        f.write(str(args))

    # Create model 加载模型，查看加载的方式
    print(f"Creating {args.arch}")
    # 获取models中名为arg.arch值的对象，也就是我们需要的模型

    model_maker = getattr(models, args.arch)
    # model_maker为我们需要的模型的类，然后使用构造函数去构造模型
    model = model_maker(
        4, 3,
        width=args.width, deep_supervision=args.deep_sup,
        norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)

    print(f"total number of trainable parameters {count_parameters(model)}")

    if args.swa:
        # Create the average model
        swa_model = model_maker(
            4, 3,
            width=args.width, deep_supervision=args.deep_sup,
            norm_layer=get_norm_layer(args.norm_layer))
        for param in swa_model.parameters():
            param.detach_()
        swa_model = swa_model.cuda()
        swa_model_optim = WeightSWA(swa_model)

    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    if args.devices == 'cpu':
        pass
    else:
        model = model.cuda()
    print(model)
    model_file = args.save_folder / "model.txt"
    with model_file.open("w") as f:
        print(model, file=f)

    criterion = Fuse_Loss()
    metric = [criterion.EDice.metric,criterion.mse.metric]
    # metric = criterion.EDice.metric
    print(metric)
 
    rangered = False  # needed because LR scheduling scheme is different for this optimizer
    optimizer = None
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay, eps=1e-4)

    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9,
                                    nesterov=True)

    elif args.optim == "adamw":
        print(f"weight decay argument will not be used. Default is 11e-2")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    elif args.optim == "ranger":
        optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        rangered = True

    if args.debug:
        args.epochs = 2
        args.warm = 0
        args.val = 1

    # full 全量其余交叉验证
    if args.full:
        # 获得三个数据集，训练集，验证集，和基准数据集，基准数据集与验证集无异
        train_dataset, bench_dataset,val_dataset,bins = get_datasets(args.seed, args.debug, full=True,val=args.val_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)

        bench_loader = torch.utils.data.DataLoader(
            bench_dataset, batch_size=args.batch_size, num_workers=args.workers)
        val_loader=torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)

    else:
        train_dataset, bench_dataset,val_dataset  ,bins= get_datasets(args.seed, args.debug, fold_number=args.fold)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True,
            pin_memory=False, num_workers=args.workers)

        bench_loader = torch.utils.data.DataLoader(
            bench_dataset, batch_size=args.batch_size, num_workers=args.workers)
        print("Val dataset number of batch:", len(val_loader))
    # print(str(bins))
    print("Train dataset number of batch:", len(train_loader))

    # create grad scaler 损失缩放加快训练

    scaler = GradScaler()

    # Actual Train loop

    best_loss = np.inf
    best_acc = -np.inf
    best_acc2 = -np.inf
    best_mse = np.inf
    print("start warm-up now!")
    if args.warm != 0:
        tot_iter_train = len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda cur_iter: (1 + cur_iter) / (tot_iter_train * args.warm))

    patients_perf = []
    # 不是第一次之前有检查点就自动warm
    # if not args.resume:
    #     for epoch in range(args.warm):
    #         ts = time.perf_counter()
            
    #         model.train()

    #         training_loss ,_= step(train_loader, model, criterion, metric, args.deep_sup, optimizer, epoch, t_writer,
    #                              scaler, scheduler, save_folder=args.save_folder,
    #                              no_fp16=args.no_fp16, patients_perf=patients_perf,bins=bins,weight=weight)
    #         te = time.perf_counter()
    #         print(f"Train Epoch done in {te - ts} s")

    #         # Validate at the end of epoch every val step
    #         if (epoch + 1) % args.val == 0 and not args.full:
    #             model.eval()
    #             with torch.no_grad():
    #                 validation_loss,_ = step(val_loader, model, criterion, metric, args.deep_sup, optimizer, epoch,
    #                                        t_writer, save_folder=args.save_folder,
    #                                        no_fp16=args.no_fp16)

    #             t_writer.add_scalar(f"SummaryLoss/overfit_sum", validation_loss[0] - training_loss[0], epoch)
    #             t_writer.add_scalar(f"SummaryLoss/overfit_mse", validation_loss[1] - training_loss[1], epoch)
    #             t_writer.add_scalar(f"SummaryLoss/overfit_ed", validation_loss[2] - training_loss[2], epoch)
    #             t_writer.add_scalar(f"SummaryLoss/overfit_bin", validation_loss[3] - training_loss[3], epoch)

    if args.warm_restart:
        print('Total number of epochs should be divisible by 30, else it will do odd things')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,args.epochs+30)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)
    if args.resume:
        bins=reload_ckpt(args, model, optimizer, scheduler=scheduler)
        # reload_ckpt(args, model, optimizer, scheduler=scheduler)
        # bins=torch.Tensor( [179.,  177.,  153.,  318., 1172.])


    print("start training now!")
    if args.swa:
        # c = 15, k=3, repeat = 5
        c, k, repeat = 30, 3, args.swa_repeat
        epochs_done = args.epochs
        reboot_lr = 0
        if args.debug:
            c, k, repeat = 2, 1, 2
    epoch_count=  args.epochs + args.warm
    if args.val_test:
        epoch_count=1; 

    for epoch in range(args.start_epoch,epoch_count):
        try:

            # do_epoch for one epoch
            ts = time.perf_counter()
            if epoch> 350:
                model,weight= freeze_model(model,type_freeze="seg")
            weight=torch.Tensor([0.01,2e-7,1]).cuda().float()
            if epoch>100:
                weight=torch.Tensor([0.001,5e-7,1]).cuda().float()
            # if epoch<20:
                
            if args.val_test==False:
                model.train()
                training_loss,_,pre_out= step(train_loader, model, criterion, metric, args.deep_sup, optimizer, epoch, t_writer,
                                    scaler, save_folder=args.save_folder,
                                    no_fp16=args.no_fp16, patients_perf=patients_perf,bins=bins,weight=weight,model_name=args.arch)
                te = time.perf_counter()
                print(f"Train Epoch done in {te - ts} s")
                if epoch>350 and epoch%20==0:
                    train_loader,val_loader,bench_loader= update_data_set(pre_out,epoch,args)

            # Validate at the end of epoch every val step
            if ((epoch + 1) % args.val == 0) or args.val_test:
                model.eval()
                with torch.no_grad():
                    if args.test:
                        validation_loss,metric_res,_ = step(val_loader, model, criterion, metric, args.deep_sup, optimizer,
                                           epoch,
                                           t_writer,
                                           save_folder=args.save_folder,
                                           no_fp16=args.no_fp16, patients_perf=patients_perf,bins=bins,weight=weight
                                           ,model_name=args.arch,test=True,args=args)
                    else:
                        validation_loss,metric_res,_ = step(val_loader, model, criterion, metric, args.deep_sup, optimizer,
                                           epoch,
                                           t_writer,
                                           save_folder=args.save_folder,
                                           no_fp16=args.no_fp16, patients_perf=patients_perf,bins=bins,weight=weight
                                           ,model_name=args.arch,args=args)
                if args.val_test or args.test:
                    return
                t_writer.add_scalar(f"SummaryLoss/overfit_sum", validation_loss[0] - training_loss[0], epoch)
                t_writer.add_scalar(f"SummaryLoss/overfit_mse", validation_loss[1] - training_loss[1], epoch)
                t_writer.add_scalar(f"SummaryLoss/overfit_ed", validation_loss[2] - training_loss[2], epoch)
                t_writer.add_scalar(f"SummaryLoss/overfit_bin", validation_loss[3] - training_loss[3], epoch)

                if validation_loss[0] < best_loss:
                    best_loss = validation_loss[0]
                    model_dict = model.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch, arch=args.arch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            bins=bins
                        ),
                        save_folder=args.save_folder, modul_name="model_best_loss.pth.tar")
                    with open(args.save_folder / "best",'a') as f:
                        f.write(f"epoch:{epoch}  best_loss {str(validation_loss)} mse:{metric_res[1]} acc:{metric_res[2]}")
                if metric_res[1] < best_mse:
                    best_mse = metric_res[1]
                    model_dict = model.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch, arch=args.arch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            bins=bins
                        ),
                        save_folder=args.save_folder,modul_name="model_best_mse.pth.tar" )
                    with open(args.save_folder / "best",'a') as f:
                        f.write(f"epoch:{epoch}  best_mse mse:{metric_res[1]} acc:{metric_res[2]}\n")
                if metric_res[2] > best_acc:
                    best_acc = metric_res[2]
                    model_dict = model.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch, arch=args.arch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            bins=bins
                        ),
                        save_folder=args.save_folder,modul_name="model_best_acc.pth.tar" )
                    with open(args.save_folder / "best",'a') as f:
                        f.write(f"epoch:{epoch}  best_acc  mse:{metric_res[1]} acc:{metric_res[2]}\n")
                if metric_res[3] > best_acc2:
                    best_acc2 = metric_res[3]
                    model_dict = model.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch, arch=args.arch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            bins=bins
                        ),
                        save_folder=args.save_folder,modul_name="model_best_acc2.pth.tar" )
                    with open(args.save_folder / "best",'a') as f:
                        f.write(f"epoch:{epoch}  best_acc  mse:{metric_res[1]} acc:{metric_res[2]} acc2:{metric_res[3]}\n")
                
                    
                save_checkpoint(
                    dict(
                        epoch=epoch, arch=args.arch,
                        state_dict=model_dict,
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        bins=bins
                    ),
                    save_folder=args.save_folder,modul_name="model_resumer.pth.tar" )
                with open(args.save_folder / "resumer",'a') as f:
                    f.write(f"epoch:{epoch}  best_acc  mse:{metric_res[1]} acc:{metric_res[2]}")
                
                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")
            model_dict = model.state_dict()
            save_checkpoint(
                    dict(
                        epoch=epoch, arch=args.arch,
                        state_dict=model_dict,
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        bins=bins
                    ),
                    save_folder='/mntcephfs/lab_data/wangcm/wqs/brats/open_brats2020/runs',modul_name="model_resumer.pth.tar" )

            if args.swa:
                if (args.epochs - epoch - c) == 0:
                    reboot_lr = optimizer.param_groups[0]['lr']

            if not rangered:
                scheduler.step()
                print("scheduler stepped! "+"")
            else:
                if epoch / args.epochs > 0.5:
                    scheduler.step()
                    print("scheduler stepped!")

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            model_dict = model.state_dict()
            save_checkpoint(
                dict(
                    epoch=epoch, arch=args.arch,
                    state_dict=model_dict,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    bins=bins
                ),
                save_folder=args.save_folder,modul_name="moduel_resume.pth" )

            break

    if args.swa:
        swa_model_optim.update(model)
        print("SWA Model initialised!")
        for i in range(repeat):
            optimizer = torch.optim.Adam(model.parameters(), args.lr / 2, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c + 10)
            for swa_epoch in range(c):
                # do_epoch for one epoch
                ts = time.perf_counter()
                model.train()
                swa_model.train()
                current_epoch = epochs_done + i * c + swa_epoch
                training_loss,_,_ = step(train_loader, model, criterion, metric, args.deep_sup, optimizer,
                                     current_epoch, t_writer,
                                     scaler, no_fp16=args.no_fp16, patients_perf=patients_perf,bins=bins)
                te = time.perf_counter()
                print(f"Train Epoch done in {te - ts} s")

                t_writer.add_scalar(f"SummaryLoss/train_sum", training_loss[0], current_epoch)
                t_writer.add_scalar(f"SummaryLoss/train_mse", training_loss[1], current_epoch)
                t_writer.add_scalar(f"SummaryLoss/train_ed", training_loss[2], current_epoch)
                t_writer.add_scalar(f"SummaryLoss/train_bin", training_loss[3], current_epoch)

                # update every k epochs and val:
                print(f"cycle number: {i}, swa_epoch: {swa_epoch}, total_cycle_to_do {repeat}")
                if (swa_epoch + 1) % k == 0:
                    swa_model_optim.update(model)
                    if not args.full:
                        model.eval()
                        swa_model.eval()
                        with torch.no_grad():
                            validation_loss,_,_ = step(val_loader, model, criterion, metric, args.deep_sup, optimizer,
                                                   current_epoch,
                                                   t_writer, save_folder=args.save_folder, no_fp16=args.no_fp16)
                            swa_model_loss,_ ,_= step(val_loader, swa_model, criterion, metric, args.deep_sup, optimizer,
                                                  current_epoch,
                                                  t_writer, swa=True, save_folder=args.save_folder,
                                                  no_fp16=args.no_fp16)

                        t_writer.add_scalar(f"SummaryLoss/val", validation_loss, current_epoch)
                        t_writer.add_scalar(f"SummaryLoss/swa", swa_model_loss, current_epoch)
                        t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss, current_epoch)
                        t_writer.add_scalar(f"SummaryLoss/overfit_swa", swa_model_loss - training_loss, current_epoch)
                scheduler.step()
        epochs_added = c * repeat
        save_checkpoint(
            dict(
                epoch=args.epochs + epochs_added, arch=args.arch,
                state_dict=swa_model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                bins=bins
            ),
            save_folder=args.save_folder, )
    else:
        save_checkpoint(
            dict(
                epoch=args.epochs, arch=args.arch,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                bins=bins
            ),
            save_folder=args.save_folder, )

    try:
        df_individual_perf = pd.DataFrame.from_records(patients_perf)
        print(df_individual_perf)
        df_individual_perf.to_csv(f'{str(args.save_folder)}/patients_indiv_perf.csv')
        reload_ckpt_bis(f'{str(args.save_folder)}/model_best.pth.tar', model)
        generate_segmentations(bench_loader, model, t_writer, args)
    except KeyboardInterrupt:
        print("Stopping right now!")


def step(data_loader, model, criterion, metric, deep_supervision, optimizer, epoch, writer, scaler=None,
         scheduler=None, swa=False, save_folder=None, no_fp16=False, patients_perf=None, sur=True,cpu=True,
         bins=[1,1],weight=[0.1,0.5,1],model_name="Unet",test=False,args=None,p_label=False):
    # Setup
    bins=bins.cuda()
    # no_fp16=True
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses1 = AverageMeter('sum', ':.4e')
    losses2 = AverageMeter('mse', ':.4e')
    losses3 = AverageMeter('ed', ':.4e')
    losses4 = AverageMeter('type', ':.4e')
    # TODO monitor teacher loss
    mode = "train" if model.training else "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses2,losses3,losses4],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics1 = []
    metrics2 = []
    metrics3 = []
    metrics4 = []
    print(f"fp 16: {not no_fp16}")
    # TODO: not recreate data_aug for each epoch...
    data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cuda()
    sur_days=[]
    sur_types=[]
    sur_ids=[]
    alives=[]
    pseudos=[]

    for i, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        # batch['sur_label'] = torch.stack(batch['sur_label'])
        sur_target_label=torch.Tensor(batch["sur_label"]).cuda()
        targets = batch["label"].cuda(non_blocking=True)
        inputs = batch["image"].cuda()
        age=batch["age"].cuda().unsqueeze(1)
        sex=batch["sex"].cuda().unsqueeze(1)
        p_id=batch["patient_id"]
        race=batch["race"].cuda().unsqueeze(1)
        type=batch["type"].cuda().unsqueeze(1)
        inputs2 = torch.cat((age, sex, type,race), dim=1).to(torch.float32)
        inputs=[inputs,inputs2]
        sur_target = batch["sur_days"].cuda()
        alive = batch["alive"].cuda()
        pseudo = batch["pseudo"].cuda()
        patient_id = batch["patient_id"]

        with autocast(enabled=not no_fp16):
            # data augmentation step
            if mode == "train":
                inputs[0] = data_aug(inputs[0])
                # inputs[1] = data_aug(inputs[1])
            if deep_supervision:
                segs, deeps = model(inputs[0],inputs[1])
                if mode == "train":  # revert the data aug
                    segs, deeps = data_aug.reverse([segs, deeps])

                loss_ = torch.stack(
                    [criterion(segs, targets)] + [criterion(deep, targets) for
                                                  deep in deeps])
                print(f"sum:{loss_[0]},mse:{loss_[1]},ed:{loss_[2]},bin:{loss_[3]}")
                loss_ = torch.mean(loss_[0])
            else:
                if sur and model_name=='Unet' :
                    segs, sur_out = model(inputs[0],inputs[1])
                elif sur and model_name=='Unet2':
                    segs ,sur_out,type_out = model(inputs[0],inputs[1])
                elif sur and model_name=='Unet3':
                    segs ,sur_out,type_out = model(inputs[0])
                    # print(sur_out.size())
                if mode == "train":
                    segs = data_aug.reverse(segs)
                if sur and model_name=='Unet2' and test==False :
                    loss_ = criterion(segs,targets,sur_out,sur_target,type_out,sur_target_label,alive,bins,pseudo,weight=weight)
                elif test==False:
                    loss_ = criterion(segs,targets,sur_out,sur_target,None,sur_target_label,alive,bins,pseudo,weight=weight)
            if patients_perf is not None and not test:
                patients_perf.append(
                    dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_[0].item())
                )

            

            # measure accuracy and record loss_
            if not test and not np.isnan(loss_[0].item())  :
                losses1.update(loss_[0].item())
                losses2.update(loss_[1].item())
                losses3.update(loss_[2].item())
                losses4.update(loss_[3].item())
            else:
                print("NaN in model loss!!")

            if not model.training and not test:
                
                metric_1 = metric[0](segs, targets)
                metrics1.extend(metric_1)
                # segs      , targets   , sur_out   , sur_target, sur_target_label, alive, bins, pseudo
                # seg_input , seg_target, sur_inputs, sur_target, sur_target_label, alive, bins, pseudo
                # sur_inputs, sur_target, sur_target_label, alive,pseudo, bins
                metric_2 ,metric_3,metric_4= metric[1](sur_out,sur_target,type_out,sur_target_label,alive,pseudo,bins)
                metrics2.extend(metric_2)
                metrics3.extend(metric_3)
                metrics4.extend(metric_4)
        
        if test:
            with open(args.save_folder / "sur_val",'a') as f:
                    for i in range(sur_out.size()[0]):
                        f.write(f"{p_id[i]},{sur_out[i].item(), type_out[i].argmax(dim=0)}\n")
        else:
            writer.add_scalar(f"Loss/{mode}{'_swa' if swa else ''}",
                              loss_[0].item(),
                              global_step=batch_per_epoch * epoch + i)
            # compute gradient and do SGD step
            # print(segs)
        if model.training and loss_[0].item()!=0:
            # print(loss_[0])
            scaler.scale(loss_[0]).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)
            print(optimizer.param_groups[0]['lr'],"  lr")
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)
        # segs ,sur_out,type_out
        sur_days.append(sur_out.detach().cpu().numpy())
        if model_name=="Unet2":
            sur_types.append(type_out.detach().cpu().numpy())
        sur_ids.append(p_id)
        alives.append(alive.detach().cpu().numpy())
        pseudos.append(pseudo.detach().cpu().numpy())
    metrics=None
    if not model.training and not test:
        metrics= save_metrics(epoch, [metrics1,metrics2,metrics3,metrics4], swa, writer, epoch, False, save_folder)
        # save_metrics(epoch, metrics2, swa, writer, epoch, False, save_folder+"2")
    
    if mode == "train":
        writer.add_scalar(f"SummaryLoss/train_sum", losses1.avg, epoch)
        writer.add_scalar(f"SummaryLoss/train_msi", losses2.avg, epoch)
        writer.add_scalar(f"SummaryLoss/train_ed", losses3.avg, epoch)
        writer.add_scalar(f"SummaryLoss/train_bin", losses4.avg, epoch)
    elif not test:
        writer.add_scalar(f"SummaryLoss/val_sum", losses1.avg, epoch)
        writer.add_scalar(f"SummaryLoss/val_msi", losses2.avg, epoch)
        writer.add_scalar(f"SummaryLoss/val_ed", losses3.avg, epoch)
        writer.add_scalar(f"SummaryLoss/val_bin", losses4.avg, epoch)

    return [losses1.avg,losses2.avg,losses3.avg,losses4.avg],metrics,[sur_ids,sur_days,sur_types,alives,pseudos]


def freeze_model(model, type_freeze="suf"):
    if type_freeze == "seg":
        for k, v in model.named_parameters():

            if k in ["encoder1","encoder2","encoder3","encoder4","bottom", "bottom_2","downsample",\
                     "gap", "decoder3", "decoder2", "decoder1","upsample","outconv"]:
                v.requires_grad = False
            else:
                v.requires_grad = True
        weight=torch.Tensor([0.1,1,0])
        print("freeze seg")
    if type_freeze == "suf":
        for k, v in model.named_parameters():
            if k in ["linear1", "linear2", "linear3", "linear4", "linear5"]:
                v.requires_grad = False
            else:
                v.requires_grad = True
        weight=torch.Tensor([0.1,0,1])
        print("freeze suf")
    if type_freeze == "no":
        for k, v in model.named_parameters():
                v.requires_grad = True
        weight=torch.Tensor([0.1,0.5,1])
        print("no freeze ")
    return model,weight
    


def un_freeze_model(model):
    for k, v in model.named_parameters():
        v.requires_grad = True
    return model


def update_data_set(pre_out,epoch,args):
    print("start pseudos label")
    pseudos=np.concatenate(pre_out[4])
    pseudos_mark=np.where((pseudos==0)|(pseudos==1))
    sur_ids=np.concatenate(pre_out[0])[pseudos_mark]
    sur_days=np.concatenate(pre_out[1])[pseudos_mark]
    sur_types=np.concatenate(pre_out[2],axis=0)[pseudos_mark]
    # alives=np.concatenate(pre_out[3])[pseudos_mark]
    c=Confidence()
    confid=c.get_confi(sur_days,sur_types)
    confid_mark=np.where(confid<0.32)
    if confid_mark[0].size!=0:
        # return 
        confid=confid[confid_mark[0]]
        confids_days=sur_days[confid_mark[0]]
        confids_ids=sur_ids[confid_mark[0]]
        print(f"eopch{epoch} get confidence count: {len(confids_days)} mean: {confid.mean()} \n")
        with open(args.save_folder / "confids",'a') as f:
            f.write(f"eopch{epoch} get confidence count: {len(confids_days)} mean: {confid.mean()} \n")
            for i in range(len(confid)):
                f.write(f"{confids_ids[i]} : {confids_days[i]}\n")
        df = pd.read_csv(BRATS_SUR_FOLDER, encoding="utf-8",index_col='id')
        for i in range(len(confids_ids)):
            df.loc[confids_ids[i],'pseudo']=1
            df.loc[confids_ids[i],'sur_days']=confids_days[i]
        df.to_csv(BRATS_SUR_FOLDER)
    if args.full:
        # 获得三个数据集，训练集，验证集，和基准数据集，基准数据集与验证集无异
        train_dataset, bench_dataset,val_dataset,bins = get_datasets(args.seed, args.debug, full=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)

        bench_loader = torch.utils.data.DataLoader(
            bench_dataset, batch_size=1, num_workers=args.workers)
        val_loader=torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)

    else:
        train_dataset, val_dataset, bench_dataset ,bins= get_datasets(args.seed, args.debug, fold_number=args.fold)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True,
            pin_memory=False, num_workers=args.workers, collate_fn=determinist_collate)

        bench_loader = torch.utils.data.DataLoader(
            bench_dataset, batch_size=1, num_workers=args.workers)
        print("Val dataset number of batch:", len(val_loader))
    return train_loader,val_loader,bench_loader





if __name__ == '__main__':
    arguments = parser.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    # torch.cuda.set_device(1)
    main(arguments)
