import tensorflow as tf
import time
import os
import numpy as np
import cv2

import utils

from training import model
from training import loss


def get_grid_imgs(g_in, g_out, batch_size, z_dim, sess, z_test=None):    
    g_imgs = []
    if 16 > batch_size:        
        assert 16 % batch_size == 0, 'batch size is not 2^n'
        for i in range(16//batch_size):
            if z_test is None:
                z = np.random.normal( size=[batch_size, z_dim] )
            else:
                z = z_test[i*batch_size:(i+1)*batch_size]
            fake_imgs = sess.run(g_out, feed_dict={g_in:z})
            g_imgs.append(fake_imgs)
    else:
        if z_test is None:
            z = np.random.normal( size=[batch_size, z_dim] )
        else:
            z = z_test
        fake_imgs = sess.run(g_out, feed_dict={g_in:z})[:16]
        g_imgs.append(fake_imgs)
        

    g_imgs = np.array(g_imgs)
    group, minibatch, h, w, c = g_imgs.shape
    b = group*minibatch

    g_imgs = np.reshape(g_imgs, (b, h, w, c))

    rows = 4
    cols = 4
    g_imgs = ((g_imgs+1)*127.5).astype(np.uint8)

    grid_img = np.zeros((h*rows, w*cols, c), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = g_imgs[row*cols + col]

    return grid_img


def train(dataset_dir, result_dir, batch_size, epochs, lr, load_path):
    # make results dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    files = os.listdir(result_dir)
    results_number = len(files)

    #set dir name
    desc = 'DCGAN'
    desc +='_batch-%d'%batch_size
    desc +='_epoch-%d'%epochs

    save_dir = os.path.join(result_dir, '%04d_' %(results_number) + desc)
    os.mkdir(save_dir)

    #ckpt dir
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    os.mkdir(ckpt_dir)

    # set my logger
    log = utils.my_log(os.path.join(save_dir, 'results.txt'))
    log.logging('< Info >')


    # load data
    imgs = utils.load_images_in_folder(dataset_dir)
    imgs_num = imgs.shape[0]
    log.logging('dataset path : ' + dataset_dir)
    log.logging('results path : '+ save_dir)
    log.logging('load model path : '+ str(load_path))
    log.logging('load images num : %d' %(imgs_num))
    log.logging('image shape : (%d, %d, %d)' %(imgs.shape[1],imgs.shape[2],imgs.shape[3]))

    
    # images preprocessing [-1 , 1]
    imgs = (imgs - 127.5)/127.5
    

    ### train setting
    np.random.seed(2222)
    z_dim = 100
    beta_1 = 0.5
    log.logging('z dim : %d' %z_dim)

    # input placeholder
    g_in = tf.placeholder(tf.float32, shape=(None, z_dim))
    d_in = tf.placeholder(tf.float32, shape=(None, imgs.shape[1],imgs.shape[2],imgs.shape[3]))

    y_true = tf.placeholder(tf.float32, shape=(None, 1))

    # set model
    g_out = model.Generator(g_in, batch_size)
    d_out = model.Discriminator(d_in)
    d_out_g_in = model.Discriminator(model.Generator(g_in, batch_size))

    # set loss                               
    g_loss = loss.loss(y_true, d_out_g_in)   
    d_loss = loss.loss(y_true, d_out)

    # set trainable variables
    all_var = tf.trainable_variables()
    g_vars = [var for var in all_var if 'Generator' in var.name]
    d_vars = [var for var in all_var if 'Discriminator' in var.name]

    # show trainable parameters
    '''for var in g_vars:
        print(var)
    for var in d_vars:
        print(var)'''

    # set optimizer
    G_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta_1, name='Adam_G').minimize(g_loss, var_list=g_vars)
    D_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta_1, name='Adam_D').minimize(d_loss, var_list=d_vars)
    log.logging('G opt : Adam(lr:%f, beta_1:%f)' %(lr, beta_1))
    log.logging('D opt : Adam(lr:%f, beta_1:%f)' %(lr, beta_1))
    log.logging('total epoch : %d' %epochs)
    log.logging('batch size  : %d' %batch_size)
    
    log.logging(utils.SPLIT_LINE, log_only=True)

    # train 
    log.logging('< train >')
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        if load_path is None:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, load_path)
            log.logging('['+load_path+'] model loaded !!')

        train_start_time = time.time()

        #test noise z
        z_test = np.random.normal(size=(batch_size, z_dim))

        for epoch in range(1, epochs+1):
            start_epoch_time = time.time()
            epoch_g_loss = 0
            epoch_d_loss = 0
            # remaining data is not used
            for step in range(imgs_num//batch_size): 
                
                # G and D inputsz_test
                z = np.random.normal( size=[batch_size, z_dim] )
                real_imgs = imgs[step*batch_size:(step+1)*batch_size]
                fake_imgs = sess.run(g_out, feed_dict={g_in:z})
                
                # training D
                sess.run(D_opt, feed_dict={d_in:fake_imgs, y_true:np.zeros((batch_size, 1))})
                sess.run(D_opt, feed_dict={d_in:real_imgs, y_true:np.ones((batch_size, 1))})

                # training G
                z = np.random.normal( size=[batch_size, z_dim] )
                sess.run(G_opt, feed_dict={g_in:z, y_true:np.ones((batch_size, 1))})

                # calculate loss
                loss_g = sess.run(g_loss, feed_dict={g_in:z, y_true:np.ones((batch_size, 1))})

                loss_d_real = sess.run(d_loss, feed_dict={d_in:real_imgs, y_true:np.ones((batch_size, 1))})
                loss_d_fake = sess.run(d_loss, feed_dict={d_in:fake_imgs, y_true:np.zeros((batch_size, 1))})
                loss_d = loss_d_real + loss_d_fake

                epoch_g_loss += loss_g
                epoch_d_loss += loss_d

                #print('%03d / %03d loss (G : %f || D : %f ) detail:(f:%f r:%f)' %(step, imgs_num//batch_size, loss_g, loss_d, loss_d_fake, loss_d_real) )
            
            epoch_g_loss /= (imgs_num//batch_size) 
            epoch_d_loss /= (imgs_num//batch_size) 
            
            log.logging('[%d/%d] epoch << G loss: %.5f || D loss: %.5f >>  time: %.1f sec' %(epoch, epochs, epoch_g_loss, epoch_d_loss, time.time()-start_epoch_time))

            # make fake imgs
            fake_imgs = get_grid_imgs(g_in, g_out, batch_size, z_dim, sess, z_test)
            cv2.imwrite(os.path.join(save_dir, 'fake %05depoc.png' %epoch), fake_imgs)

            # model save
            save_name = os.path.join(ckpt_dir, 'model.ckpt')
            saver.save(sess, save_name, global_step=epoch)
        
        log.logging('\n[%d] epoch finish! << fianl G loss: %.5f || final D loss: %.5f >>  total time: %.1f sec' %(epochs, epoch_g_loss, epoch_d_loss, time.time()-train_start_time))

    print('train finished!')


def validation(load_path, result_dir, generate_num, seed):
    # make results dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    files = os.listdir(result_dir)
    results_number = len(files)

    #set dir name
    desc = 'DCGAN'
    desc +='_generate'
    save_dir = os.path.join(result_dir, '%04d_' %(results_number) + desc)
    os.mkdir(save_dir)

    # set my logger
    log = utils.my_log(os.path.join(save_dir, 'results.txt'))
    log.logging(utils.SPLIT_LINE)
    log.logging('< Validation >')

    # set seed
    np.random.seed(seed) #default 22222

    # validation set 
    z_dim = 100

    # input placeholder
    g_in = tf.placeholder(tf.float32, shape=(None, z_dim))

    # set model
    g_out = model.Generator(g_in, 1)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, load_path)
        log.logging('['+load_path+'] model loaded !!')

        for idx in range(generate_num):
            z = np.random.normal( size=[1, z_dim] )
            fake_img = sess.run(g_out, feed_dict={g_in:z})[0]

            # pixel intensity : -1 ~ 1 -> 0 ~ 255
            fake_img = ((fake_img + 1) * 127.5).astype(np.uint8)
            log.logging('[%d/%d] Generate image!! [seed:%d]' %(idx, generate_num, seed))
            cv2.imwrite(os.path.join(save_dir, 'fake%04d_seed%d.png' %(idx, seed)), fake_img)

        



