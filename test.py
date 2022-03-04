import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import time

from dataset import DeepFashionDataset
from model import Generator, Discriminator, VGGLoss, Discriminator_feat

try:
    import wandb
except ImportError:
    wandb = None


from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def getFace(images, FT, LP, RP):
    """
    images: are images where we want to get the faces
    FT: transform to get the aligned face
    LP: left pad added to the imgae
    RP: right pad added to the image
    """
    faces = []
    b, h, w, c = images.shape
    for b in range(images.shape[0]):
        if not (abs(FT[b]).sum() == 0): # all 3x3 elements are zero
            # only apply the loss to image with detected faces
            # need to do this per image because images are of different shape
            current_im = images[b][:, :, int(RP[b].item()):w-int(LP[b].item())].unsqueeze(0)
            theta = FT[b].unsqueeze(0)[:, :2] #bx2x3
            grid = torch.nn.functional.affine_grid(theta, (1, 3, 112, 96))
            current_face = torch.nn.functional.grid_sample(current_im, grid)
            faces.append(current_face)
    if len(faces) == 0:
        return None
    return torch.cat(faces, 0)


def generate(args, loader, g_ema, device):



    g_l1 = torch.tensor(0.0, device=device)
    g_vgg = torch.tensor(0.0, device=device)
    loss_dict = {}

    criterionL1 = torch.nn.L1Loss()
    criterionVGG = VGGLoss(device).to(device)

    for i, data in enumerate(loader):
        batch_start_time = time.time()

        input_image = data['input_image'].float().to(device)
        real_img = data['target_image'].float().to(device)
        flow = data['flow'].float().to(device)
        sil = data['target_sil'].float().to(device)
        pred_img_ori = data['pred_image'].float().to(device)
        attention = data['attention'].float().to(device)
        attention = F.interpolate(attention, size=(args.size, args.size), mode='bilinear', align_corners=True)
        sil = F.interpolate(sil, size=(args.size, args.size))
        source_sil = data['input_sil'].float().to(device)
        #todo flow 배경부분에 왜 값이 있는지 확인 0.0039정도 있음
        flow = F.interpolate(flow, args.size)

        if source_sil.size(1) == 3:
            source_sil = source_sil[:,:1]

        if sil.size(1) == 3:
            sil = sil[:,:1]

        if pred_img_ori.size(3) == 512:
            pred_img = pred_img_ori.clone()
            pred_img_ori = F.interpolate(pred_img_ori, size=(256,256))
        elif pred_img_ori.size(3) == 256:
            pred_img = F.interpolate(pred_img_ori, size=(args.size, args.size), mode='nearest')
        else:
            raise NotImplementedError()

        # mask out the padding
        # else only focus on the foreground - initial step of training

        real_img = real_img * sil

        # appearance = human foregound + fg mask (pass coor for warping)

        # complete_coor = data['complete_coor'].float().to(device)
        # if args.size == 256:
        #     complete_coor = torch.nn.functional.interpolate(complete_coor, size=(256, 256), mode='bilinear')
        warped_img = F.grid_sample(input_image, flow.permute(0,2,3,1)) * sil

        if args.finetune:
            appearance = torch.cat([input_image, source_sil], 1)
        else:
            if args.allview:
                appearance = torch.cat([input_image * source_sil, source_sil, attention], 1)
            else:
                appearance = torch.cat([input_image * source_sil, source_sil], 1)

        with torch.no_grad():
            g_ema.eval()
            sample, _ = g_ema(appearance=appearance, flow=flow,
                              sil=sil, input_feat=pred_img_ori,
                              condition=pred_img
                              )

            save_path = os.path.join(args.path, 'output_stage2', '/'.join([args.ckpt.split('/')[-2], args.ckpt.split('/')[-1][:-3]]))
            for j in range(sample.size(0)):
                os.makedirs(os.path.join(save_path, data['model_id'][j]), exist_ok=True)

                utils.save_image(
                    (sample[j] /2 + 0.5) * sil[j] + (1-sil[j]),
                    os.path.join(os.path.join(save_path, data['model_id'][j]), f"{data['source_view_id'][j]}_{data['target_view_id'][j]}.png"),
                )

                utils.save_image(
                    (input_image[j] / 2 + 0.5) * source_sil[j] + (1-source_sil[j]), #  * (1- )
                    os.path.join(os.path.join(save_path, data['model_id'][j]),
                                 f"{data['source_view_id'][j]}.png"),
                )



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Pose with Style trainer")

    parser.add_argument("--path", type=str, default='/home/nas1_temp/dataset/Thuman', help="path to the lmdb dataset")
    parser.add_argument("--batch", type=int, default=4, help="batch sizes for each gpus")
    parser.add_argument("--workers", type=int, default=4, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=4, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=512, help="image sizes for the model")
    parser.add_argument("--vol_feat_res", type=int, default=128, help="resolution of volume feature")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--finetune", action="store_true", help="finetune to handle background- second step of training.")
    parser.add_argument("--allview", action="store_true")
    # parser.add_argument("--gpu_ids", type=str, default=0, help="add face loss when faces are detected")

    args = parser.parse_args()

    args.latent = 2048
    args.n_mlp = 8


    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"])

    dataset = DeepFashionDataset(args.path, 'test', args.size, args.allview, args.vol_feat_res)

    loader = data.DataLoader(

        dataset,
        batch_size=args.batch,
        drop_last=False,
        pin_memory=True,
        num_workers=args.workers,
        shuffle=False,
    )


    generate(args, loader, g_ema, device)
