import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import cv2
import time
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import glob
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch
import models_mae
from models import *
from datasets import *

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--data_path", type=str, default="./jittor_landscape_200k")
parser.add_argument("--output_path", type=str, default="./results/flickr")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--input_hight", type=int, default=192, help="size of image height")
parser.add_argument("--input_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
parser.add_argument('--resume', default='../pretrained/mae_visualize_vit_base.pth',
                        help='resume from checkpoint')
opt = parser.parse_args()
print(opt)

def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)
    return img

os.makedirs(f"{opt.output_path}/images/", exist_ok=True)
os.makedirs(f"{opt.output_path}/saved_models/", exist_ok=True)

# Configure dataloaders
transforms = [
    transform.Resize(size=(int(opt.input_hight*1.5), int(opt.input_width*1.5)), mode=Image.BICUBIC),
    transform.RandomCrop((opt.input_hight, opt.input_width)),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

val_transforms = [
    transform.Resize(size=(int(opt.input_hight), int(opt.input_width)), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]



path = sorted(glob.glob(os.path.join(opt.data_path, 'train', "labels") + "/*.*"))
img_id_table = [each.split('/')[-1][:-4] for each in path]

writer = SummaryWriter(opt.output_path)

# Loss functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixelwise = nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.input_hight // 2 ** 4, opt.input_width // 2 ** 4)

# Initialize generator and discriminator
model = models_mae.__dict__[opt.model](norm_pix_loss=False, img_styles = len(img_id_table)) 
model_without_ddp = model 
# print("Model = %s" % str(model_without_ddp)) 

param_groups = optim_factory.add_weight_decay(model_without_ddp, 0.05, skip_list=('cls_token')) 
optimizer_G = torch.optim.AdamW(param_groups, lr=opt.lr, betas=(0.9, 0.95))
loss_scaler = NativeScaler() 
# discriminator = Discriminator()
misc.load_model(args=opt, model_without_ddp=model_without_ddp, optimizer=optimizer_G, loss_scaler=loss_scaler) 

generator = model

dataloader = ImageDataset(opt.data_path, mode="train", opt=opt).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

train_samples = ImageDataset(opt.data_path, mode="train_sample", opt=opt).set_attrs(
    batch_size=10,
    shuffle=False,
    num_workers=1,
)

val_dataloader = ImageDataset(opt.data_path, mode="eval", opt=opt).set_attrs(
    batch_size=10,
    shuffle=False,
    num_workers=2,
)

@jt.single_process_scope()
def eval(epoch, writer):
    generator.eval()
    cnt = 1
    os.makedirs(f"{opt.output_path}/images/trainSamples/epoch_{epoch}", exist_ok=True)
    for i, (real_img, real_label, img_ids, idxs) in enumerate(train_samples):
        real_label_tensor = torch.tensor(np.array(real_label),device ='cuda', requires_grad=True)
        img_idx = torch.tensor([img_id_table.index(ids) for ids in img_ids]).cuda()
        # img_idx = torch.tensor(np.array(idxs),device ='cuda')
        fake_img_tensor = generator(real_label_tensor, img_ids= img_idx)

        fake_B = generator.unpatchify(fake_img_tensor)
        
        if i == 0:
            # visual image result
            img_sample = np.concatenate([real_label.data, fake_B.cpu().data, real_img.data], -2)
            img = save_image(img_sample, f"{opt.output_path}/images/epoch_{epoch}_train_sample.png", nrow=5)
            # writer.add_image('val/image', img.transpose(2,0,1), epoch)
            
            # Check style token similarity
            
            assert len(img_idx) == 10
            val_img_cls_token = torch.index_select(generator.cls_token, 0, img_idx)
            simi_potos = []
            for k in val_img_cls_token:
                most_simi = 0.
                temp_img = ''
                for ind in range(len(img_id_table)):
                    if ind in img_idx:
                        continue
                    j = torch.index_select(generator.cls_token, 0, torch.tensor([ind]).cuda())
                    simi = (torch.softmax(j,-1)*torch.softmax(k,-1)).sum()
                    if simi>most_simi:
                        most_simi = simi
                        temp_img = img_id_table[ind]
                simi_potos.append(temp_img)
            simi_log = ', '.join(str(i) for i in simi_potos)
            writer.add_text('Most similiar photos: ', simi_log, epoch)
            print('Most similiar photos: ',simi_log)
            
            transforms = [
                transform.Resize(size=(int(opt.input_hight), int(opt.input_width)), mode=Image.BICUBIC)
            ]
            trs = transform.Compose(transforms)
            listo = []
            col_num = 2
            for index in range(len(simi_potos)//col_num):
                listi = []
                for i in range(col_num):
                    label_path = os.path.join(opt.data_path, "train", "imgs", simi_potos[(index*col_num+i) % len(simi_potos)]) + ".jpg"
                    img = Image.open(label_path)
                    img = trs(img)
                    img = np.array(img)[:, :, ::-1]
                    listi.append(img)
                img = np.concatenate(listi,axis=0)
                listo.append(img)
            img = np.concatenate(listo,axis=1)    
            cv2.imwrite(f"{opt.output_path}/images/epoch_{epoch}_train_similiar_style.png",img)


        fake_B = fake_B.cpu().data
        for idx in range(fake_B.shape[0]):
            path = f"{opt.output_path}/images/trainSamples/epoch_{epoch}/{img_ids[idx]}.jpg"
            save_image(fake_B[idx].unsqueeze(0).numpy(), path, nrow=1)
            cnt += 1

#  evaluate
    cnt = 1
    os.makedirs(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}", exist_ok=True)
    for i, (_, real_label, img_ids, _) in enumerate(val_dataloader):
        img_idx = torch.tensor([random.randint(0,len(img_id_table)-1) for ids in img_ids]).cuda()

        real_label_tensor = torch.tensor(np.array(real_label),device ='cuda', requires_grad=True)
        fake_img_tensor = generator(real_label_tensor, img_ids= img_idx)

        fake_B = generator.unpatchify(fake_img_tensor)
        
        if i == 0:
            # visual image result
            img_sample = np.concatenate([real_label.data, fake_B.cpu().data], -2)
            img = save_image(img_sample, f"{opt.output_path}/images/epoch_{epoch}_test.png", nrow=5)
            
        fake_B = fake_B.cpu().data
        for idx in range(fake_B.shape[0]):
            path = f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}/{img_ids[idx]}.jpg"
            save_image(fake_B[idx].unsqueeze(0).numpy(), path, nrow=1)
            cnt += 1



warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------
generator.to('cuda')


class jt2tor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, jt_grad):
        ctx.save_for_backward(jt_grad,input)
   
        return input

    @staticmethod
    def backward(ctx, grad_output):
      
        jt_grad,input = ctx.saved_tensors
        d_input = grad_output*jt_grad
        # d_weight = input*grad_output
    
        return d_input, None


def restore_loss(target, pred):
   
    loss = (pred - target) ** 2
    loss = loss.mean()  # [N, L], mean loss per patch

    return loss

prev_time = time.time()
eval(0, writer)
for epoch in range(opt.epoch, opt.n_epochs):
    generator.train()
    for i, (real_img, real_label, img_ids, idxs) in enumerate(dataloader):
        # Adversarial ground truths
        valid = jt.ones([real_label.shape[0], 1]).stop_grad()
        fake = jt.zeros([real_label.shape[0], 1]).stop_grad()
        real_label_tensor = torch.tensor(np.array(real_label),device ='cuda', requires_grad=True)
        real_img_tensor = torch.tensor(np.array(real_img),device ='cuda', requires_grad=True)
        optimizer_G.zero_grad()
        img_idx = torch.tensor(np.array(idxs),device ='cuda')
        fake_img_tensor = generator(real_label_tensor, img_ids=img_idx)
        
        fake_img_tensor = generator.unpatchify(fake_img_tensor)
        assert fake_img_tensor.shape == real_img_tensor.shape
        loss_G = restore_loss(real_img_tensor, fake_img_tensor)
        loss_G.backward()
        
        optimizer_G.step()
        writer.add_scalar('train/loss_G', loss_G.item(), epoch * len(dataloader) + i)

        jt.sync_all(True)

        if jt.rank == 0:
            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            jt.sync_all()
            if batches_done % 5 == 0:
                
                print(
                    "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_G.cpu().data,
                        time_left,
                    )   
                )

    if jt.rank == 0 and opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        eval(epoch, writer)
        # Save model checkpoints
    if epoch == opt.n_epochs-1 or epoch == 0:
        to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer_G.state_dict(),
            'epoch': epoch,
            'args': opt,
        }
        save_dir=os.path.join(f"{opt.output_path}/saved_models/generator_{epoch}.pkl")
        torch.save(to_save, save_dir)
        # discriminator.save(os.path.join(f"{opt.output_path}/saved_models/discriminator_{epoch}.pkl"))
