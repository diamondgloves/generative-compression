# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session

import tensorflow as tf
import numpy as np
import os, time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import moxing as mox
    mox.file.shift('os', 'mox')
except ImportError:
    print('package moxing not imported')

class Utils(object):
    @staticmethod
    def conv_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu):
        in_kwargs = {'center':True, 'scale': True}
        x = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
        x = tf.contrib.layers.instance_norm(x, **in_kwargs)
        x = actv(x)

        return x

    @staticmethod
    def upsample_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu, training=True, batch_norm=False):
        bn_kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        in_kwargs = {'center':True, 'scale': True}
        x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
        if batch_norm is True:
            x = tf.layers.batch_normalization(x, **bn_kwargs)
        else:
            x = tf.contrib.layers.instance_norm(x, **in_kwargs)
        x = actv(x)
        return x
    # def upsample_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu):
    #     in_kwargs = {'center':True, 'scale': True}
    #     x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
    #     x = tf.contrib.layers.instance_norm(x, **in_kwargs)
    #     x = actv(x)
    #     return x
    
    @staticmethod
    def residual_block(x, n_filters, kernel_size=3, strides=1, actv=tf.nn.relu):
        # init = tf.contrib.layers.xavier_initializer()
        strides = [1,1]
        identity_map = x

        p = int((kernel_size-1)/2)
        res = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                activation=None, padding='VALID')
        res = actv(tf.contrib.layers.instance_norm(res))

        res = tf.pad(res, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                activation=None, padding='VALID')
        res = tf.contrib.layers.instance_norm(res)

        assert res.get_shape().as_list() == identity_map.get_shape().as_list(), 'Mismatched shapes between input/output!'
        out = tf.add(res, identity_map)

        return out

    @staticmethod
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    @staticmethod
    def scope_variables(name):
        with tf.variable_scope(name):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    @staticmethod
    def run_diagnostics(model, config, args, sess, saver, train_handle, start_time, epoch, name, G_loss_best, D_loss_best, step):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_diagnosis = {model.training_phase: False, model.handle: train_handle}

        G_loss, D_loss, summary = sess.run([model.G_loss, model.D_loss, model.merge_op], feed_dict=feed_dict_diagnosis)
        model.train_writer.add_summary(summary, step)
        # except tf.errors.OutOfRangeError:
        #     G_loss, D_loss = float('nan'), float('nan')

        if G_loss < G_loss_best and D_loss < D_loss_best:
            G_loss_best, D_loss_best = G_loss, D_loss
            improved = '[*]'
            if step > 20000 :
                if not os.path.exists(args.ckpt_best):
                    os.makedirs(args.ckpt_best)
                save_path = saver.save(sess, os.path.join(args.ckpt_best, '{}_step{}.ckpt'.format(name, step)), global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))
        if step % 20000 == 0 and step > 20000 and step % config.diagnostic_steps != 0:
            if not os.path.exists(os.getcwd()+'/checkpoints/'):
                os.makedirs(os.getcwd()+'/checkpoints/')
            save_path = saver.save(sess, os.path.join(os.getcwd(), 'checkpoints', '{}_step{}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))
            # if args.is_cloud:
            #     mox.file.copy(save_path, os.path.join(args.train_url, '{}_step{}.ckpt'.format(name, epoch)))
        print('Iteration {} | Epoch {} | Generator Loss: {:.3f} | Discriminator Loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(step, epoch, G_loss, D_loss, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

        return G_loss_best, D_loss_best

    @staticmethod
    def single_plot(epoch, global_step, sess, model, handle, name, samples_path, single_compress=False, is_cloud=False):

        real = model.example
        gen = model.reconstruction

        # Generate images from noise, using the generator network.
        if single_compress:
            r, g = sess.run([real, gen], feed_dict={model.training_phase:False, model.handle: handle})
        else:
            r, g = sess.run([real, gen], feed_dict={model.training_phase:True, model.handle: handle})

        images = list()

        for im, imtype in zip([r,g], ['real', 'gen']):
            im = ((im+1.0))/2  # [-1,1] -> [0,1]
            im = np.squeeze(im)
            im = im[:,:,:3]
            images.append(im)

            # Uncomment to plot real and generated samples separately
            # f = plt.figure()
            # plt.imshow(im)
            # plt.axis('off')
            # f.savefig("{}/gan_compression_{}_epoch{}_step{}_{}.pdf".format(samples_path, name, epoch,
            #                     global_step, imtype), format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
            # plt.gcf().clear()
            # plt.close(f)
        
        # print(images[0].max(axis=(0,1)),images[0].min(axis=(0,1)),images[1].max(axis=(0,1)), images[1].min(axis=(0,1)))
        # print(images[0].mean(axis=(0,1)),images[0].var(axis=(0,1)),images[1].mean(axis=(0,1)), images[1].var(axis=(0,1)))
        # print(images[0].shape,images[1].shape)
        # input()
        # from PIL import Image
        # png = tf.image.convert_image_dtype(images[1], dtype=tf.uint16)
        # png = Image.fromarray(tf.Session().run(png))
        # png.save(name)

        comparison = np.hstack(images)
        f = plt.figure()
        plt.imshow(comparison)
        plt.axis('off')
        if single_compress:
            f.savefig(name, format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
            # f.savefig(name, dpi=720, bbox_inches='tight', pad_inches=0)
        else:
            if is_cloud:
                fig_file = os.path.join(os.getcwd(), 'samples/','{}_epoch{}_step{}_{}_comparison.pdf'.format(name, epoch, global_step, imtype))
                if not os.path.exists(os.path.dirname(fig_file)):
                    os.makedirs(os.path.dirname(fig_file))
                f.savefig(fig_file, format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
                mox.file.copy(fig_file, os.path.join(samples_path, os.path.basename(fig_file)))
            else:
                if not os.path.exists(samples_path):
                    os.makedirs(samples_path)
                f.savefig("{}/{}_epoch{}_step{}_{}_comparison.pdf".format(samples_path, name, epoch,
                    global_step, imtype), format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
            
        plt.gcf().clear()
        plt.close(f)


    @staticmethod  # not used
    def weight_decay(weight_decay, var_label='DW'):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'{}'.format(var_label)) > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(weight_decay, tf.add_n(costs))

    @staticmethod
    def mox_copy_with_timeout_retry(src_data, dst_data, retry_num, timeout, file_or_not):
        """
        Use this Func to substitude moxi.file.copy and moxi.file.copy_parallel.
        `("s3://wolfros-net/datasets/imagenet.tar", "/cache/imagenet.tar", 4, 3600, "True") -> None`
        Warning:
        this method depends on the file download.py which should be in the same directory of this file!!
        """
        status = 0
        # set the cmd string to excute
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download.py')
        cmd = 'timeout %(timeout)s python %(path)s %(src)s %(dst)s %(file_or_not)s' % {
            'timeout': timeout, 'src': src_data, 'dst': dst_data, 'path': path, 'file_or_not': file_or_not}
        print(cmd)
        for i in range(0, retry_num):
            ret = os.system(cmd)
            if ret == 0:
                print('copy success')
                status = 1
                break
            print('ret: %d retry' % (i + 1))
        if status != 1:
            print('copy fail exit')
            return 1
        else:
            return 0
