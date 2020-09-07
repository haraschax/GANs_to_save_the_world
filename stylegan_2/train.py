#!/usr/bin/env python
import os
import argparse
import torch
import torch.multiprocessing as mp
from retry.api import retry_call
from tqdm import tqdm
from helpers import NanException
from stylegan2_pytorch import Trainer

from datetime import datetime

if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default=gpu_count, type=int)
    parser.add_argument('-m', '--master', default='localhost', type=str)
    args = parser.parse_args()


def train_from_folder(
    gpu,
    data='./data',
    results_dir='./results',
    models_dir='./models', 
    name='default',
    new=True,
    load_from=-1,
    image_size = 256,
    network_capacity = 16,
    transparent = False,
    batch_size = 5,
    gradient_accumulate_every = 6,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    lr_mlp = 0.1,
    ttur_mult = 1.5,
    num_workers =  None,
    save_every = 1000,
    generate = False,
    generate_interpolation = False,
    save_frames = False,
    num_image_tiles = 8,
    trunc_psi = 0.75,
    fp16 = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [],
    no_const = False,
    aug_prob = 0.,
    dataset_aug_prob = 0.,
):
    using_ddp = True
    model = Trainer(
        name,        
        results_dir,
        models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        transparent = transparent,
        lr = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        num_workers = num_workers,
        save_every = save_every,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        dataset_aug_prob = dataset_aug_prob,
        using_ddp=using_ddp,
        gpu=gpu
    )
    if using_ddp:
        world_size = torch.cuda.device_count()
        torch.distributed.init_process_group(backend='nccl', rank=gpu, world_size=world_size)
    else:
        gpu = 0
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)


    if not new:
        model.load(load_from)
    else:
        model.clear()

    if generate:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f'generated-{timestamp}'
        model.evaluate(samples_name, num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if generate_interpolation:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f'generated-{timestamp}'
        model.generate_interpolation(samples_name, num_image_tiles, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    model.set_data_src(data, using_ddp=using_ddp)

    for _ in tqdm(range(num_train_steps - model.steps), mininterval=10., desc=f'{name}<{data}>'):
        model.train()#retry_call(model.train, tries=3, exceptions=NanException)
        if _ % 50 == 0:
            model.print_log()

if __name__ == "__main__":
  #  fire.Fire(train_from_folder)
    world_size = torch.cuda.device_count()
    print(world_size)
    if False:#args.gpus == 1 and args.nodes == 1:
        print("running single process")
        train_from_folder(None, args)
    else:
        os.environ['MASTER_ADDR'] = args.master
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train_from_folder,
                 args=[],
                 nprocs=world_size,
                 join=True)
