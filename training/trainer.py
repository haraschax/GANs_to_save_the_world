import dataloader as DL
from config import config
import network as net
from math import floor, ceil
import os
import sys
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
import tf_recorder as tensorboard
import utils as utils
import numpy as np
from torchsummary import summary
# import tensorflow as tf


class trainer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')

        self.nz = config.nz
        self.ni = config.ni
        self.optimizer = config.optimizer

        self.resl = 7           # we start from 2^2 = 4
        self.lr = config.lr
        self.eps_drift = config.eps_drift
        self.smoothing = config.smoothing
        self.max_resl = config.max_resl
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.globalIter = 0
        self.globalTick = 0
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen':None, 'dis':None}
        self.complete = {'gen':0, 'dis':0}
        self.phase = 'init'
        self.flag_flush_gen = False
        self.flag_flush_dis = False
        self.flag_add_noise = self.config.flag_add_noise
        self.flag_add_drift = self.config.flag_add_drift

        # network and cirterion
        self.G = net.Generator(config)
        self.D = net.Discriminator(config)
        print ('Generator structure: ')
        print(self.G)
        print ('Discriminator structure: ')
        print(self.D)
        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        if self.use_cuda:
            self.mse = self.mse.cuda()
            torch.cuda.manual_seed(config.random_seed)
            if config.n_gpu==1:
                self.G = self.G.cuda(device=0)
                self.D = self.D.cuda(device=0)
            else:
                gpus = []
                for i in range(config.n_gpu):
                    gpus.append(i)
                self.G = torch.nn.DataParallel(self.G, device_ids=gpus).cuda()
                self.D = torch.nn.DataParallel(self.D, device_ids=gpus).cuda()


        # define tensors, ship model to cuda, and get dataloader.
        self.renew_everything()

        # tensorboard
        self.use_tb = config.use_tb
        if self.use_tb:
            self.tb = tensorboard.tf_recorder()



    def renew_everything(self):
        # renew dataloader.
        self.loader = DL.dataloader(config)
        self.loader.renew(min(floor(self.resl), self.max_resl))

        # define tensors
        self.z = torch.FloatTensor(self.loader.batchsize, self.nz)
        self.coord = torch.FloatTensor(self.loader.batchsize, self.ni)
        self.coord_test = torch.FloatTensor(16, self.ni)
        self.x_test_true = torch.FloatTensor(16, 3, self.loader.imsize, self.loader.imsize)
        self.x_test = torch.FloatTensor(16, 3, self.loader.imsize, self.loader.imsize)
        self.x = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.x_tilde = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.real_label = torch.FloatTensor(self.loader.batchsize).fill_(1)
        self.fake_label = torch.FloatTensor(self.loader.batchsize).fill_(0)

        # enable cuda
        if self.use_cuda:
            self.z = self.z.cuda()
            self.coord = self.coord.cuda()
            self.x = self.x.cuda()
            self.x_tilde = self.x.cuda()
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()
            torch.cuda.manual_seed(config.random_seed)

        # wrapping autograd Variable.
        self.x = Variable(self.x)
        self.x_tilde = Variable(self.x_tilde)
        self.z = Variable(self.z)
        self.coord = Variable(self.coord)
        self.real_label = Variable(self.real_label)
        self.fake_label = Variable(self.fake_label)
        
        # ship new model to cuda.
        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
        
        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
        

    def feed_interpolated_input(self, x):
        if self.phase == 'gtrns' and floor(self.resl)>2 and floor(self.resl)<=self.max_resl:
            alpha = self.complete['gen']/100.0
            transform = transforms.Compose( [   transforms.ToPILImage(),
                                                transforms.Scale(size=int(pow(2,floor(self.resl)-1)), interpolation=0),      # 0: nearest
                                                transforms.Scale(size=int(pow(2,floor(self.resl))), interpolation=0),      # 0: nearest
                                                transforms.ToTensor(),
                                            ] )
            x_low = x.clone().add(1).mul(0.5)
            for i in range(x_low.size(0)):
                x_low[i] = transform(x_low[i]).mul(2).add(-1)
            x = torch.add(x.mul(alpha), x_low.mul(1-alpha)) # interpolated_x

        if self.use_cuda:
            return x.cuda()
        else:
            return x

    def add_noise(self, x):
        # TODO: support more method of adding noise.
        if self.flag_add_noise==False:
            return x

        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + torch.mean(self.fx_tilde).item() * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        z = np.random.randn(*x.size()).astype(np.float32) * strength
        z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
        return x + z

    def add_loc_noise(self, x):
        # TODO: support more method of adding noise.
        if self.flag_add_noise==False:
            return x

        strength = 0.5
        z = np.random.randn(*x.size()).astype(np.float32) * strength
        z[:,2] = 0
        z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
        return x + z

    def train(self):
        #self.coord_test = torch.FloatTensor(self.loader.batchsize, self.ni)
        #self.coord_test = Variable(self.coord_test, volatile=True)
        #self.coord_test.data.resize_(self.loader.batchsize, self.ni).uniform_(-90.0, 90.0)
        #self.coord_test[:,1] = torch.mul(self.coord_test[:,1], 2)
        #self.coord_test[:,2] = torch.floor(torch.remainder(self.coord_test[:,2], 7)).add(1)
        #np.savetxt('repo/coord_test.txt', self.coord_test.cpu().numpy())
        test_batch = self.loader.get_batch()
        self.coord_test.data[:8] = test_batch['meta'].cuda()
        self.x_test_true.data[:8] = self.feed_interpolated_input(test_batch['image'])
        test_batch = self.loader.get_batch()
        self.coord_test.data[8:] = test_batch['meta'].cuda()
        self.x_test_true.data[8:] = self.feed_interpolated_input(test_batch['image'])
        self.z_test = np.random.randn(self.coord_test.size()[0], 512).astype(np.float32)
        self.z_test = Variable(torch.from_numpy(self.z_test)).cuda() if self.use_cuda else Variable(torch.from_numpy(self.z_test))

        g_losses, l1_metrics, d_losses = [], [] ,[]
        for epoch in range(5000): #True: #step in range(2, self.max_resl+1+5):
            self.lr = config.lr_decay*self.lr
            for g in self.opt_d.param_groups:
              g['lr'] = self.lr
            for g in self.opt_g.param_groups:
              g['lr'] = self.lr

            for step in range(128):
                self.globalIter = self.globalIter+1

                # zero gradients.
                self.G.zero_grad()
                self.D.zero_grad()

                # update discriminator.
                batch = self.loader.get_batch()
                self.x.data = self.feed_interpolated_input(batch['image'])
                #if self.flag_add_noise:
                #    self.x = self.add_noise(self.x)
                self.coord.data = batch['meta'].cuda()
                self.fx = self.D(self.x, self.coord)

                batch = self.loader.get_batch()
                self.coord.data = batch['meta'].cuda()
                self.x.data = self.feed_interpolated_input(batch['image'])
                z = np.random.randn(self.coord.size()[0], 512).astype(np.float32)
                z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
                self.x_tilde = self.G(self.coord, z)
                self.fx_tilde = self.D(self.x_tilde.detach(), self.coord)

                loss_d = torch.mean(self.fx_tilde - self.fx) + torch.mean(.001*(torch.square(self.fx)))  #self.mse(self.fx.squeeze(), self.real_label) + self.mse(self.fx_tilde, self.fake_label)
                loss_d.backward()
                self.opt_d.step()

                # update generator.
                batch = self.loader.get_batch()
                self.coord.data = batch['meta'].cuda()
                self.x.data = self.feed_interpolated_input(batch['image'])
                z = np.random.randn(self.coord.size()[0], 512).astype(np.float32)
                z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
                self.x_tilde = self.G(self.coord, z)
                self.fx_tilde = self.D(self.x_tilde, self.coord)
                metric_l1 = self.mae(self.x, self.x_tilde)
                loss_g = torch.mean(-self.fx_tilde)
                #loss_g = self.mae(self.x, self.x_tilde)
                #loss_g = torch.mean((fx_tilde.squeeze(), self.real_label.detach()) #+ self.mae(self.x, self.x_tilde)
                loss_g.backward()
                self.opt_g.step()
                
                # logging.
                log_msg = ' [E:{0}][T:{1}][{2:6}/{3:6}]  errD: {4:.4f} | errG: {5:.4f} | {6:.5f}| {7:.8f}]'.format(epoch, step, step, 128, loss_d.item(), loss_g.item(), metric_l1.item(), self.lr)
                tqdm.write(log_msg)

                if self.use_tb:
                    self.tb.add_scalar('data/loss_g', loss_g.item(), epoch*128 + step)
                    self.tb.add_scalar('data/loss_d', loss_d.item(), epoch*128 + step)
                    self.tb.add_scalar('data/l1_metric', metric_l1.item(), epoch*128 + step)
                g_losses.append(loss_g.item())
                d_losses.append(loss_d.item())
                l1_metrics.append(metric_l1.item())



            # save model.
            #self.snapshot('repo/model', epoch)

            # tensorboard visualization.
            if self.use_tb:
                with torch.no_grad():
                    self.x_test = self.G(self.coord_test.cuda(), self.z_test)
                self.tb.add_scalar('data/loss_g_epoch', np.mean(g_losses), epoch)
                self.tb.add_scalar('data/loss_d_epoch', np.mean(d_losses), epoch)
                self.tb.add_scalar('data/l1_metric_epoch', np.mean(l1_metrics), epoch)
                self.tb.add_scalar('tick/lr', self.lr, self.globalIter)
                g_losses, l1_metrics, d_losses = [], [] ,[]
                #self.tb.add_scalar('tick/cur_resl', int(pow(2,floor(self.resl))), self.globalIter)
                #IMAGE GRID
                self.tb.add_image_grid('grid/x_test', 4, utils.adjust_dyn_range(self.x_test.data.float(), [0,1], [0,1]), epoch)
                self.tb.add_image_grid('grid/x_true', 4, utils.adjust_dyn_range(self.x_test_true.data.float(), [0,1], [0,1]), epoch)

    def get_state(self, target):
        if target == 'gen':
            state = {
                'resl' : self.resl,
                'state_dict' : self.G.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resl' : self.resl,
                'state_dict' : self.D.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state


    def snapshot(self, path, epoch):
        if not os.path.exists(path):
            if os.name == 'nt':
                os.system('mkdir {}'.format(path.replace('/', '\\')))
            else:
                os.system('mkdir -p {}'.format(path))
        # save every 100 tick if the network is in stab phase.
        #ndis = 'dis_E{}.pth.tar'.format(int(floor(self.resl)), self.globalTick)
        ngen = 'gen_E{}.pth.tar'.format(int(epoch))
        save_path = os.path.join(path, ngen)
        if not os.path.exists(save_path):
          #torch.save(self.get_state('dis'), save_path)
          #save_path = os.path.join(path, ngen)
          torch.save(self.get_state('gen'), save_path)
          print('[snapshot] model saved @ {}'.format(path))

if __name__ == '__main__':
    ## perform training.
    print('----------------- configuration -----------------')
    for k, v in vars(config).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')
    torch.backends.cudnn.benchmark = True           # boost speed.
    trainer = trainer(config)
    trainer.train()


