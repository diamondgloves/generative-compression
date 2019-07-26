#!/usr/bin/python3
try:
    import moxing as mox
    mox.file.shift('os', 'mox')
except ImportError:
    print('package moxing not imported')
import zipfile
import tarfile

import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_train

tf.logging.set_verbosity(tf.logging.ERROR)

def train(config, args):
    start_time = time.time()
    G_loss_best, D_loss_best = float('inf'), float('inf')
    if args.is_cloud:
        ckpt = tf.train.get_checkpoint_state(args.ckpt_local_dir)
    else:
        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)

    # Load data
    print('Training on dataset {}'.format(args.dataset))
    if config.use_conditional_GAN:
        print('Using conditional GAN')
        paths, semantic_map_paths = Data.load_dataframe(os.path.join(os.path.dirname(__file__),args.train_list), load_semantic_maps=True)
        test_paths, test_semantic_map_paths = Data.load_dataframe(os.path.join(os.path.dirname(__file__), args.test_list), load_semantic_maps=True)
    else:
        paths = Data.load_dataframe(os.path.join(os.path.dirname(__file__),args.train_list))
        test_paths = Data.load_dataframe(os.path.join(os.path.dirname(__file__),args.test_list))
        if args.is_cloud:
            paths = np.array([path.replace('data','/cache') for path in paths])
            test_paths = np.array([test_path.replace('data','/cache') for test_path in test_paths])

    # Build graph
    gan = Model(config, paths, name=args.name, dataset=args.dataset, tensorboard='tensorboard/')
    saver = tf.train.Saver()

    if config.use_conditional_GAN:
        feed_dict_test_init = {gan.test_path_placeholder: test_paths, gan.test_semantic_map_path_placeholder: test_semantic_map_paths}
        feed_dict_train_init = {gan.path_placeholder: paths, gan.semantic_map_path_placeholder: semantic_map_paths}
    else:
        feed_dict_test_init = {gan.test_path_placeholder: test_paths}
        feed_dict_train_init = {gan.path_placeholder: paths}
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_handle = sess.run(gan.train_iterator.string_handle())
        test_handle = sess.run(gan.test_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            # Continue training saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
        elif args.restore_meta_file:   # still not work 
            new_saver = tf.train.import_meta_graph(args.restore_meta_file)
            new_saver.restore(sess, args.restore_meta_file)
            print('{} restored.'.format(args.restore_meta_file))
        else:
            print('no model to be restored')
        # sess.run(gan.test_iterator.initializer, feed_dict=feed_dict_test_init)

        # sess.graph.finalize()

        step = 0
        tsbd_count = 1
        for epoch in range(config.num_epochs):
            sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_train_init)
            # Run diagnostics
            G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, args, sess, saver, train_handle,
                start_time, epoch, args.name, G_loss_best, D_loss_best, step)

            while True:
                # send a copy of event file from executing enviroment to S3 bucket
                if args.is_cloud and (time.time() - start_time) > tsbd_count * config.tensorboard_time_interval:
                    tsbd_path = os.popen('find {} -name {}_train_*'.format('tensorboard/', args.name)).readlines()[0][:-1]
                    Utils.mox_copy_with_timeout_retry(tsbd_path, args.tensorboard, 4, 3600, 'False')
                    tsbd_count += 1
                # train Generator and Discriminator
                try:
                    # Update generator
                    # for _ in range(4):
                    feed_dict = {gan.training_phase: True, gan.handle: train_handle}
                    sess.run(gan.G_train_op, feed_dict=feed_dict)

                    # Update discriminator 
                    step, _ = sess.run([gan.D_global_step, gan.D_train_op], feed_dict=feed_dict)
                    # print('step:{}'.format(step))

                    if step % config.diagnostic_steps == 0:
                        G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, args, sess, saver, train_handle,
                            start_time, epoch, args.name, G_loss_best, D_loss_best, step)
                        Utils.single_plot(epoch, step, sess, gan, train_handle, args.name, args.samples, is_cloud=args.is_cloud)
                        # for _ in range(4):
                        #    sess.run(gan.G_train_op, feed_dict=feed_dict)

                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    if args.is_cloud:
                        save_path = saver.save(sess, os.path.join(args.ckpt_local_dir, '{}_last.ckpt'.format(args.name)), global_step=epoch)
                        Utils.mox_copy_with_timeout_retry(os.path.dirname(save_path), args.train_url, 4, 3600, 'False')
                        print('Interrupted, model saved to: ', save_path, 'then copy to {}'.format(args.train_url))
                    else: 
                        save_path = saver.save(sess, os.path.join(args.ckpt_path, '{}_last.ckpt'.format(args.name)), global_step=epoch)
                        print('Interrupted, model saved to: ', save_path)
                    gan.train_writer.close()
                    sys.exit()
        if is_cloud:
            save_path = saver.save(sess, os.path.join(args.ckpt_local_dir, '{}_end.ckpt'.format(args.name)), global_step=epoch)
            Utils.mox_copy_with_timeout_retry(os.path.dirname(save_path), args.train_url, 4, 3600, 'False')
        else:
            save_path = saver.save(sess, os.path.join(args.ckpt_path, '{}_end.ckpt'.format(args.name)), global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} hours".format(save_path, (time.time()-start_time)/3600))

def main(**kwargs):
    # train using ModelArts

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", default="GAN-train", help="Checkpoint/Tensorboard label", type=str)
    parser.add_argument("--dataset", default="cityscapes", help="choice of training dataset. Currently only supports cityscapes/ADE20k", choices=set(("cityscapes", "ADE20k","OpenImages")), type=str)
    parser.add_argument('--train_list', default='data/cityscapes_paths_train.h5', help='path list of training dataset', type=str)
    parser.add_argument('--val_list', default='data/cityscapes_paths_val.h5', help='path list of validation dataset', type=str)
    parser.add_argument('--test_list', default='data/cityscapes_paths_test.h5', help='path list of test dataset', type=str)
    parser.add_argument("--data_tar_file", default='s3://ai-codec/s50003064/data/leftImg8bit_trainvaltest.zip', help="path to dataset .tar file", type=str)

    parser.add_argument("--is_cloud", help="run on the cloud", action="store_true")
    parser.add_argument('--tensorboard', default='tensorboard/', help='',type=str)
    parser.add_argument('--samples', default='samples/', help='generated samples during training', type=str)
    
    parser.add_argument("--restore_last", help="restore last saved model in the ckpt_path", action="store_true")
    parser.add_argument('--ckpt_path', default='', help='path where to seek the last checkpoint', type=str)
    parser.add_argument("--restore_meta_file", help="path to meta file of the appointed model to be restored", type=str)
    parser.add_argument('--ckpt_best', default='checkpoints/best/', help='path to the best model during training', type=str)
    parser.add_argument('--ckpt_local_dir', default='', help='path where to seek the last checkpoint on the ModelArts', type=str)

    # arguments generated on the ModelArtscloud platform，must be preserved
    parser.add_argument('--data_url', default='', help='数据存储位置-训练数据集', type=str)
    parser.add_argument('--train_url', default='', help='训练输出文件路径', type=str)
    parser.add_argument('--num_gpus', default=2, help='number of GPUs to use', type=int)
    args = parser.parse_args()

    if args.is_cloud:
        # Decompress the tar/zip file to cache.
        local_tar_file = '/cache/' + os.path.basename(args.data_tar_file)
        # download
        Utils.mox_copy_with_timeout_retry(args.data_tar_file, local_tar_file, 4, 3600, 'True')
        # decompress
        ext = os.path.splitext(args.data_tar_file)[-1]
        if ext == '.tar':
            f = tarfile.open(local_tar_file)
        elif ext == '.zip':
            f = zipfile.ZipFile(local_tar_file)
        else:
            print('decompress failed, tar and zip format of compressed file are supported')
            return
        f.extractall(path='/cache')
        f.close()

        # copy the checkpoint files
        # args.ckpt_best = os.path.join(os.getcwd(), args.ckpt_best)
        args.ckpt_local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/')
        if not os.path.exists(args.ckpt_local_dir):
            os.makedirs(args.ckpt_local_dir)
        if args.restore_last and args.ckpt_path:
            Utils.mox_copy_with_timeout_retry(args.ckpt_path, args.ckpt_local_dir, 4, 3600, 'False')
        elif args.restore_meta_file:
            Utils.mox_copy_with_timeout_retry(os.path.dirname(args.restore_meta_file), args.ckpt_local_dir, 4, 3600, 'False')
    # Launch training
    print('Train with the configuration：')
    for key, value in config_train.__dict__.items():
        if key[0]!='_':
            print('{}: {}'.format(key, value))
    train(config_train, args)

if __name__ == '__main__':
    main()
