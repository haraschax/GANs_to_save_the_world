import json
import os
from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
import multiprocessing
import torch.distributed as dist


import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from adamp import AdamP

import torchvision
from torchvision import transforms

from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
from road_dataloader import Dataset
from helpers import cast_list, EMA, set_requires_grad, cycle, \
                    default, loss_backwards, image_noise, \
                    slerp, gradient_penalty, raise_if_nan, \
                    noise, evaluate_in_chunks, calc_pl_lengths, is_empty
from model import Discriminator, StyleVectorizer, Generator
from functools import partial


assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

num_cores = multiprocessing.cpu_count()

# constants

EXTS = ['jpg', 'jpeg', 'png']
EPS = 1e-8



# augmentations

def random_float(lo, hi):
    return lo + (hi - lo) * random()

def random_crop_and_resize(tensor, scale):
    b, c, h, _ = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random() * delta)
    w_delta = int(random() * delta)
    cropped = tensor[:, :, h_delta:(h_delta + new_width), w_delta:(w_delta + new_width)].clone()
    return F.interpolate(cropped, size=(h, h), mode='bilinear')

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob=0., detach=False, feature_vector=None):
        if random() < prob:
            random_scale = random_float(0.75, 0.95)
            images = random_hflip(images, prob=0.5)
            images = random_crop_and_resize(images, scale=random_scale)

        if detach:
            images.detach_()

        return self.D(images, feature_vector)


class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim=512, fmap_max=512, style_depth=8, network_capacity=16, transparent=False, fp16=False, cl_reg=False, steps=1, lr=1e-4, ttur_mult=2, fq_layers=[], fq_dict_size=256, attn_layers=[], use_feats=False, lr_mlp=0.1, using_ddp=False, gpu=0):
        super().__init__()
        self.lr = lr
        self.gpu = gpu
        self.steps = steps
        self.ema_updater = EMA(0.995)
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp, use_feats=use_feats)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers, fmap_max=fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers=fq_layers, fq_dict_size=fq_dict_size, attn_layers=attn_layers,
                               transparent=transparent, fmap_max=fmap_max, use_feats=use_feats)

        self.num_layers = self.G.num_layers
        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp, use_feats=use_feats)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers)
        if using_ddp:
            self.D = DDP(self.D.cuda(), device_ids=[gpu])
            self.S = DDP(self.S.cuda(), device_ids=[gpu])
            self.SE = DDP(self.SE.cuda(), device_ids=[gpu])
            self.G = DDP(self.G.cuda(), device_ids=[gpu])
            self.GE = DDP(self.GE.cuda(), device_ids=[gpu])

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = AdamP(generator_params, lr=self.lr, betas=(0.5, 0.9))
        self.D_opt = AdamP(self.D.parameters(), lr=self.lr * ttur_mult, betas=(0.5, 0.9))

        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda()

        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        if isinstance(self.G, DDP):
            blocks = self.G.module.blocks
        else:
            blocks = self.G.blocks
        for block in blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x

class Trainer():
    def __init__(self, name, results_dir, models_dir, image_size, network_capacity, transparent=False,
                batch_size=4, mixed_prob=0.9, gradient_accumulate_every=1, lr=2e-4, lr_mlp=1., ttur_mult=2,
                num_workers=None, save_every=1000, trunc_psi=0.6, fp16=False, cl_reg=False, fq_layers=[],
                fq_dict_size=256, attn_layers=[], use_feats=False, aug_prob=0., dataset_aug_prob=0., *args, **kwargs):
        self.GAN_params = [args, kwargs]
        self.GAN = None
        self.gpu = kwargs['gpu']
        torch.manual_seed(self.gpu)

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.transparent = transparent
        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size

        self.attn_layers = cast_list(attn_layers)
        self.use_feats = use_feats
        self.aug_prob = aug_prob

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0
        self.dropout_loss = 0
        self.last_cr_loss = 0
        self.q_loss = 0

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr=self.lr, lr_mlp=self.lr_mlp, ttur_mult=self.ttur_mult,
                             image_size=self.image_size, network_capacity=self.network_capacity,
                             transparent=self.transparent, fq_layers=self.fq_layers,
                             fq_dict_size=self.fq_dict_size, attn_layers=self.attn_layers,
                             fp16=self.fp16, cl_reg=self.cl_reg, use_feats=self.use_feats, *args, **kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.attn_layers = config.pop('attn_layers', [])
        self.use_feats = config.pop('use_feats', False)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'transparent': self.transparent,
                'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'use_feats': self.use_feats}

    def set_data_src(self, folder, using_ddp=False):
        self.dataset = Dataset(folder, self.image_size, transparent=self.transparent, aug_prob=self.dataset_aug_prob)
        if using_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                            num_replicas=dist.get_world_size(),
                            rank=dist.get_rank())
        else:
            sampler = None
        self.loader = cycle(data.DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size, drop_last=True, pin_memory=True))
        i = 0
        self.test_feature_vector_batch = torch.zeros((64, 1024))
        self.test_image_batch = torch.zeros((64, 3, self.image_size, self.image_size))
        while i < 64:
          test_image_batch, test_feature_vector_batch = next(self.loader)
          self.test_feature_vector_batch[i:min(64, i + test_feature_vector_batch.shape[0])] = test_feature_vector_batch.cuda()[:min(64, i + test_feature_vector_batch.shape[0]) - i]
          self.test_image_batch[i:min(64, i + test_feature_vector_batch.shape[0])] = test_image_batch.cuda()[:min(64, i + test_feature_vector_batch.shape[0]) - i]
          i += test_feature_vector_batch.shape[0]
        ext = 'jpg' if not self.transparent else 'png'
        torchvision.utils.save_image(self.test_image_batch, str(self.results_dir / self.name / f'test_img.{ext}'), nrow=8)

    def gen_styles(self, model, feats, n=None):
      layers = self.GAN.num_layers
      if n is None:
        n = noise(self.batch_size, self.GAN.latent_dim)
      w_space = self.GAN.S(n, feats)
      styles = w_space[:,None,:].repeat(1,layers,1)
      return styles

    def gen_mixed_styles(self, model, feats):
      layers = self.GAN.num_layers
      a = self.gen_styles(model, feats)
      if random() < self.mixed_prob:
        b = self.gen_styles(model, feats)
        tt = int(torch.rand(()).numpy() * layers)
        return torch.cat([a[:, :tt], b[:, tt:]], dim=1)
      else:
        return a


    def set_priority(self):
        assert self.loader is not None, 'You must first initialize the data source with `.set_data_src(<folder of images>)`'
        _ = next(self.loader)
        self.data = next(self.loader)
        #os.system("sudo renice -n -5 -p %d" % os.getpid())


    def train(self):
        if self.GAN is None:
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda()
        total_gen_loss = torch.tensor(0.).cuda()

        batch_size = self.batch_size
        image_size = self.GAN.image_size
        aug_prob = self.aug_prob

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = False  # self.steps % 32 == 0
        apply_dropout_hack = False #self.steps % 4 == 2
        apply_cl_reg_to_generated = self.steps > 20000

        backwards = partial(loss_backwards, self.fp16)
        self.dropout = nn.Dropout(p=0.5)


        if self.GAN.D_cl is not None:
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    image_batch, feature_vector_batch = next(self.loader)

                    styles = self.gen_mixed_styles(self.GAN.S, feature_vector_batch.cuda())
                    img_noise = image_noise(batch_size, image_size)

                    generated_images = self.GAN.G(styles, img_noise)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch = next(self.loader).cuda()
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt, 0)

            self.GAN.D_opt.step()

        # train discriminator
        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()
        for i in range(self.gradient_accumulate_every):
            image_batch, feature_vector_batch = next(self.loader)

            styles = self.gen_mixed_styles(self.GAN.S, self.dropout(feature_vector_batch.cuda()))
            img_noise = image_noise(batch_size, image_size)

            generated_images = self.GAN.G(styles, img_noise)
            fake_output, fake_q_loss = self.GAN.D_aug(generated_images.clone().detach(), detach=True,
                                                      prob=aug_prob, feature_vector=self.dropout(feature_vector_batch.cuda()))

            image_batch, feature_vector_batch = next(self.loader)
            image_batch.requires_grad_()
            real_output, real_q_loss = self.GAN.D_aug(image_batch.cuda(), prob=aug_prob, feature_vector=self.dropout(feature_vector_batch.cuda()))

            divergence = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
            disc_loss = divergence

            quantize_loss = (fake_q_loss + real_q_loss).mean()
            self.q_loss = float(quantize_loss.detach().item())

            disc_loss = disc_loss + quantize_loss
            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp
            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, 1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.GAN.D_opt.step()

        # train generator
        self.GAN.G_opt.zero_grad()
        for i in range(self.gradient_accumulate_every):
            image_batch, feature_vector_batch = next(self.loader)

            styles = self.gen_mixed_styles(self.GAN.S, self.dropout(feature_vector_batch.cuda()))
            img_noise = image_noise(batch_size, image_size)

            generated_images = self.GAN.G(styles, img_noise)
            fake_output, _ = self.GAN.D_aug(generated_images, prob=aug_prob, feature_vector=self.dropout(feature_vector_batch.cuda()))
            loss = fake_output.mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss
            if apply_dropout_hack:
              n = noise(self.batch_size, self.GAN.latent_dim)
              styles1 = self.gen_styles(self.GAN.S, self.dropout(feature_vector_batch.cuda()), n=n)
              styles2 = self.gen_styles(self.GAN.S, self.dropout(feature_vector_batch.cuda()), n=n)
              self.dropout_loss = nn.L1Loss()(self.GAN.G(styles2, img_noise), self.GAN.G(styles1, img_noise))
              gen_loss += self.dropout_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, 2)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.GAN.G_opt.step()

        # calculate moving averages
        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)

        if self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors
        checkpoint_num = floor(self.steps / self.save_every)

        if any(torch.isnan(val) for val in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{checkpoint_num}')
            self.load(checkpoint_num)
            raise NanException

        # periodically save results
        self.steps += 1

        if self.GAN.gpu == 0 and self.steps % self.save_every == 0:
            self.save(checkpoint_num)

        if self.GAN.gpu==0 and (self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500)):
            self.evaluate(floor(self.steps / 1000))

        self.av = None
        #print('rest took', time() - t0)

    @torch.no_grad()
    def evaluate(self, num=0, num_image_tiles=8, trunc=1.0):
        self.GAN.eval()
        ext = 'jpg' if not self.transparent else 'png'
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size
        noi = image_noise(num_image_tiles ** 2, image_size)
        latents = noise(num_image_tiles ** 2, latent_dim)

        # regular
        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, noi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)

        # moving averages
        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)


    @torch.no_grad()
    def generate_truncated(self, S, G, latents, noi, num_image_tiles=8):
        w_space = evaluate_in_chunks(self.batch_size, S, latents, self.test_feature_vector_batch[:noi.shape[0]])
        styles = w_space[:,None,:].repeat(1,self.GAN.num_layers,1)
        generated_images = evaluate_in_chunks(self.batch_size, G, styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num=0, num_image_tiles=8, trunc=1.0, save_frames=False):
        self.GAN.eval()
        ext = 'jpg' if not self.transparent else 'png'
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size
        num_layers = self.GAN.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim)
        latents_high = noise(num_rows ** 2, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        ratios = torch.linspace(0., 8., 100)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE,
                                                       latents, n, trunc_psi=self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow=num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        print(f'G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | DL: {self.dropout_loss:.2f} | CR: {self.last_cr_loss:.2f} | Q: {self.q_loss:.2f}')

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(f'./models/{self.name}', True)
        rmtree(f'./results/{self.name}', True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {'GAN': self.GAN.state_dict()}

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.gpu}
        load_data = torch.load(self.model_name(name), map_location=map_location)

        # make backwards compatible
        if 'GAN' not in load_data:
            load_data = {'GAN': load_data}

        self.GAN.load_state_dict(load_data['GAN'])

        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])
