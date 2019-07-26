#!/usr/bin/env python3
import os
class config_train(object):
    # mode = 'gan-train'
    num_epochs = 3 # 50 for cityscapes, 3 for OpenImages
    batch_size = 1
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    # lr_decay_rate = 2e-5
    # momentum = 0.9
    # weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    # kernel_size = 3
    diagnostic_steps = 1024
    tensorboard_time_interval = 600   # send the event file to S3 bucket every 10 mins

    # # WGAN
    # gradient_penalty = True
    # lambda_gp = 10
    # weight_clipping = False
    # max_c = 1e-2
    # n_critic_iterations = 20

    # # Compression
    lambda_X = 10 #12
    channel_bottleneck = 8  # {2,4,8,16}
    sample_noise = False #True
    use_vanilla_GAN = False
    use_feature_matching_loss = True
    feature_matching_weight = 10
    upsample_dim = 256
    multiscale = True
    use_conditional_GAN = False

    # # params for wide residual network
    # conv_keep_prob = 0.5
    # n_classes = 20

class config_test(object):
    # mode = 'gan-test'
    # num_epochs = 512
    batch_size = 1
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    # # lr_decay_rate = 2e-5
    # # momentum = 0.9
    # # weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    # # kernel_size = 3
    diagnostic_steps = 256

    # # WGAN
    # gradient_penalty = True
    # lambda_gp = 10
    # weight_clipping = False
    # max_c = 1e-2
    # n_critic_iterations = 5  # diff from train mode

    # Compression
    lambda_X = 10 #12
    channel_bottleneck = 8
    sample_noise = False #True
    use_vanilla_GAN = False
    use_feature_matching_loss = True
    upsample_dim = 256
    multiscale = True
    feature_matching_weight = 10
    use_conditional_GAN = False

    # # params for wide residual network
    # conv_keep_prob = 0.5
    # n_classes = 20
