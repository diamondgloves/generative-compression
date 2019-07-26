#!/bin/sh
# python train.py --name=GAN-0725_mem_explosion --ckpt_path=checkpoints --ckpt_best=checkpoints/best/
# python train.py --name=GAN-07-19_OpenImages --dataset=OpenImages --train_list=data/OpenImages_train.h5 --test_list=data/OpenImages_test.h5 --ckpt_path=checkpoints --ckpt_best=checkpoints/best/ --restore_meta_file=./checkpoints/best/Cityscapes/GAN-07-10_1_epoch36.ckpt-36.meta
# python train.py --name=GAN-07-19_OpenImages --dataset=OpenImages --train_list=data/OpenImages_train.h5 --test_list=data/OpenImages_test.h5 --restore_last --ckpt_path=checkpoints --ckpt_best=checkpoints/best/
python train.py --name=GAN-07-26_OpenImages --dataset=OpenImages --train_list=data/OpenImages_train.h5 --test_list=data/OpenImages_test.h5 --ckpt_path=checkpoints --ckpt_best=checkpoints/best/