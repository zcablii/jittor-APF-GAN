"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import os
from tensorboardX import SummaryWriter
import torch
import misc
from tqdm import tqdm
# parse options
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
opt = TrainOptions().parse()
ngpus = torch.cuda.device_count()
print(f'n_gpus={ngpus}')
print(f'gpu_ids={opt.gpu_ids}')
opt.lr = opt.lr * opt.batchSize * ngpus / 24
if not opt.distributed:
    opt.batchSize = opt.batchSize * ngpus
    opt.nThreads = opt.nThreads * ngpus

if misc.is_main_process():
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))
else:
    writer = None
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

print_sample_num = 8
for epoch in iter_counter.training_epochs():
    if opt.distributed:
        dataloader.sampler.set_epoch(epoch)
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)
        # Visualizations
        losses = trainer.get_latest_losses()
        loss_names = ['GAN','GAN_Feat','VGG','D_Fake','D_real']
        ct = 0
        for k, v in losses.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            if misc.is_main_process():
                writer.add_scalar(loss_names[ct], v, (epoch-1) * len(dataloader) + i)
                ct+=1

        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([# ('input_label', data_i['label'][:print_sample_num]),
                                   ('synthesized_image', trainer.get_latest_generated()[:print_sample_num]),
                                   ('real_image', data_i['image'][:print_sample_num])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving() and misc.is_main_process():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if (epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs) and misc.is_main_process():
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
