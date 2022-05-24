import heapq
import torch
from utils import save_network
from collections import OrderedDict
from visualizer import Visualizer
from tensorboardX import SummaryWriter
import os
# Based on torch.utils.trainer.Trainer code.

class Trainer(object):

    def __init__(self,
                params,
                 D,
                 G,
                 D_loss,
                 G_loss,
                 optimizer_d,
                 optimizer_g,
                 dataset,
                 dataiter,
                 semantic_nc = 29,
                 D_training_repeats=1,  # trainer
                 tick_nimg_default=10 * 1000,  # trainer
                 resume_nimg=0):
        self.params = params
        self.D = D
        self.G = G
        self.D_loss = D_loss
        self.G_loss = G_loss
        self.D_training_repeats = D_training_repeats
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g
        self.semantic_nc = semantic_nc
        self.dataiter = dataiter
        self.dataset = dataset
        self.cur_nimg = resume_nimg
        self.tick_start_nimg = self.cur_nimg
        self.tick_duration_nimg = tick_nimg_default
        self.iterations = 0
        self.cur_tick = 0
        self.time = 0
        self.stats = {
            'kimg_stat': { 'val': self.cur_nimg / 1000., 'log_epoch_fields': ['{val:8.3f}'], 'log_name': 'kimg' },
            'tick_stat': { 'val': self.cur_tick, 'log_epoch_fields': ['{val:5}'], 'log_name': 'tick'}
        }
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            's':    [],
            'end':  []
        }
        self.FloatTensor = torch.cuda.FloatTensor
        self.cur_batchsize = 0
        self.visualizer = Visualizer(params)
        self.generated = None
        self.data_img = None
        self.print_sample_num = 8
        self.writer = SummaryWriter(params['result_dir'], params['exp_name'])

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for (duration, unit) in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        # if queue_name == 'epoch':
        #     print('args', args)
        #     print('time ', time, 'queue_name ',queue_name)
        #     print('self.plugin_queues   ', self.plugin_queues)
        #     print('queue    ', queue)
        #     print('queue[0][0]  ', queue[0][0])
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            # if queue_name == 'epoch':
            #     print('    plugin ', plugin)
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        
        data['label'] = data['label'].cuda()
        data['instance'] = data['instance'].cuda()
        data['image'] = data['image'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.semantic_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        return input_semantics, data['image']


    def train_one_epoch(self, total_kimg, need_visual):

        # Calculate loss and optimize
        d_losses = [0, 0, 0]
        # get real images
        print_inf = True
        batch_amount = len(self.dataiter)
        
        for i, data in enumerate(self.dataiter):

            # print(self.cur_tick, i,self.cur_nimg, total_kimg * 1000, self.tick_start_nimg) # total_kimg * 1000 = 300 0000
            input_semantics , real_image = self.preprocess_input(data)
            input_semantics = input_semantics.cuda()
            if print_inf:
                # print('real_image.shape: ',real_image.shape)
                # print('need_visual', need_visual)
                print_inf = False
            real_image = real_image.cuda()
            self.cur_batchsize = real_image.size(0)
            self.cur_nimg += self.cur_batchsize
            for j in range(self.D_training_repeats):
                # calculate loss
                # print(input_semantics.shape)
                d_losses = self.D_loss(self.D, self.G, real_image, input_semantics)
                d_losses = tuple(d_losses)
                D_loss = d_losses[0]
                D_loss.backward()
                # backprop through D
                self.optimizer_d.step()
                # get new fake latents for next iterations or the generator
                # in the original implementation if separate_funcs were True, generator optimized on different fake_latents
          
            g_losses, generated = self.G_loss(self.G, self.D, input_semantics, i==batch_amount-2 and need_visual)
            if i==batch_amount-2 and need_visual:
                self.generated = generated
                self.data_img = real_image


            if type(g_losses) is list:
                g_losses = tuple(g_losses)
            elif type(g_losses) is not tuple:
                g_losses = (g_losses,)
            G_loss = g_losses[0]
            G_loss.backward()
            self.optimizer_g.step()

            self.iterations += 1
            self.writer.add_scalar('G_loss', G_loss, self.cur_nimg)
            self.writer.add_scalar('D_Fake', d_losses[2].mean(), self.cur_nimg)
            self.writer.add_scalar('D_real', d_losses[1].mean(), self.cur_nimg)
            self.writer.add_scalar('D_loss', d_losses[0], self.cur_nimg)

            self.call_plugins('iteration', self.iterations, *(g_losses + d_losses))

            if self.cur_nimg >= self.tick_start_nimg + self.tick_duration_nimg or self.cur_nimg >= total_kimg * 1000:
                print('finish epoch ', self.cur_tick)
                self.cur_tick += 1
                self.tick_start_nimg = self.cur_nimg
                self.stats['kimg_stat']['val'] = self.cur_nimg / 1000.
                self.stats['tick_stat']['val'] = self.cur_tick
                self.call_plugins('epoch', self.cur_tick)


    def run(self, total_kimg=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)
  
        while self.cur_nimg < total_kimg * 1000:
            need_visual = False
            if (self.cur_tick+1)%5==0:
                need_visual = True
            self.train_one_epoch(total_kimg, need_visual)

            if need_visual:
                visuals = OrderedDict([('synthesized_image', self.generated[:self.print_sample_num]),
                                    ('real_image', self.data_img[:self.print_sample_num])])
                self.visualizer.display_current_results(visuals, self.cur_tick)
            if (self.cur_tick)%10==0:
                save_network(self.D, 'D', self.cur_tick, self.params)
                save_network(self.G, 'G', self.cur_tick, self.params)
                pass

            
        # self.call_plugins('end', 1)