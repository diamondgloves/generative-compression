#!/usr/bin/python3
import tensorflow as tf
import numpy as np
# import pandas as pd
import os, time
import argparse

# User-defined
from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_test
import msssim

tf.logging.set_verbosity(tf.logging.ERROR)

def single_compress(config, args):
    ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
    assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'

    if config.use_conditional_GAN:
        print('Using conditional GAN')
        paths, semantic_map_paths = np.array([args.image_path]), np.array([args.semantic_map_path])
    else:
        paths = np.array([args.image_path])

    gan = Model(config, paths, name='single_compress', dataset=args.dataset, evaluate=True)
    saver = tf.train.Saver()    

    if config.use_conditional_GAN:
        feed_dict_init = {gan.path_placeholder: paths, gan.semantic_map_path_placeholder: semantic_map_paths}
    else:
        feed_dict_init = {gan.path_placeholder: paths}
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        handle = sess.run(gan.train_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        elif args.restore_meta_file:
            new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_meta_file))
            new_saver.restore(sess, args.restore_meta_file)
            print('Previous checkpoint {} restored.'.format(args.restore_meta_file))
        else:
            print('no model to be restored，task failed!')
            return

        sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_init)
        # eval_dict = {gan.training_phase: False, gan.handle: handle}

        if args.output_path is None:
            output = os.path.splitext(os.path.basename(args.image_path))
            save_path = os.path.join(args.samples, '{}_compressed.pdf'.format(output[0]))
        else:
            save_path = args.output_path
        Utils.single_plot(0, 0, sess, gan, handle, save_path, args.samples, single_compress=True, is_cloud=False)
        print('Reconstruction saved to', save_path)
    return
    
def eval_on_dir(config, args):
    starttime = time.time()
    ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
    assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'
    if config.use_conditional_GAN:
        print('Using conditional GAN')
        test_paths, test_semantic_map_paths = Data.load_dataframe(os.path.join(os.path.dirname(__file__), args.image_path), load_semantic_maps=True)
    else:
        test_paths = Data.load_dataframe(os.path.join(os.path.dirname(__file__),args.image_path))
    
    gan = Model(config, test_paths, name='single_compress', dataset=args.dataset, evaluate=True)
    saver = tf.train.Saver()
    if config.use_conditional_GAN:
        feed_dict_test_init = {gan.test_path_placeholder: test_paths, gan.test_semantic_map_path_placeholder: test_semantic_map_paths}    
    else:
        feed_dict_test_init = {gan.test_path_placeholder: test_paths}

    MSSSIM, PSNR, SSIM, i = 0, 0, 0, 0
    test_num = test_paths.shape[0]
    print('{} images of dataset {} are involvod in the evaluation'.format(test_num, args.dataset))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # train_handle = sess.run(gan.train_iterator.string_handle())
        test_handle = sess.run(gan.test_iterator.string_handle())
        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        elif args.restore_meta_file:
            new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_meta_file))
            new_saver.restore(sess, args.restore_meta_file)
            print('Previous checkpoint {} restored.'.format(args.restore_meta_file))
        else:
            print('no model to be restored, task failed! ')
        sess.run(gan.test_iterator.initializer, feed_dict=feed_dict_test_init)
        while True:
            try:
                i += 1
                feed_dict = {gan.training_phase: False, gan.handle: test_handle}
                origin, recons = sess.run([gan.example, gan.reconstruction], feed_dict=feed_dict)
                # pixel value of origin and recons are in [-1.0, 1,0]
                this_msssim = msssim.MultiScaleSSIM(origin, recons, max_val=2)
                this_psnr = sess.run(tf.image.psnr(origin, recons, max_val=2))  # type:list
                this_ssim = [0]
                # this_ssim = sess.run(tf.image.ssim_multiscale(origin, recons, max_val=2))
                print('image: {}/{} | SSIM: {:.5f} | MS-SSIM: {:.5f} | PSNR: {:.5f}'.format(i, test_num, this_ssim[0], this_msssim, this_psnr[0]))
                MSSSIM += this_msssim
                PSNR += this_psnr[0]
                SSIM += this_ssim[0]
            except tf.errors.OutOfRangeError:
                print('finish evaluating on test set!')
                print("SSIM: {:.5f} | MS-SSIM: {:.5f} | PSNR: {:.5f} | {:.3f} seconds per image".format(SSIM / test_num, MSSSIM /test_num , PSNR / test_num, (time.time()-starttime)/test_num))
                return



def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", help="only evaluate metrics", action="store_true")
    parser.add_argument("--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument('--ckpt_path', help='path to seek the last model', type=str)
    parser.add_argument("--restore_meta_file", help="path to model to be restored", type=str)
    parser.add_argument("--image_path", help="path to image to compress", type=str)
    parser.add_argument("--semantic_map_path", help="path to corresponding semantic map", type=str)
    parser.add_argument("--output_path", help="path to output image", type=str)
    parser.add_argument("--dataset", default="cityscapes", help="choice of training dataset. Currently only supports cityscapes/ADE20k", choices=set(("cityscapes", "ADE20k")), type=str)
    parser.add_argument('--samples', default='samples/', help='generated samples during training', type=str)
    args = parser.parse_args()

    if args.eval:
        eval_on_dir(config_test, args)
    else:
        if os.path.isdir(args.image_path):
            files = os.listdir(args.image_path)
            for img in files:
                if os.path.splitext(img)[-1] == '.png':
                    args.image_path = os.path.join(args.image_path, img)
                    single_compress(config_test, args)
        else:          
            single_compress(config_test, args)

if __name__ == '__main__':
    main()
