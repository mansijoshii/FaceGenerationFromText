import os
import pickle
import random
import time
import argparse

from stage1_model import generate_c, build_ca_model, build_adversarial_model, build_embedding_compressor_model, build_stage1_generator, build_stage1_discriminator
from stage1_log import save_rgb_img, write_log
from stage1_loss import KL_loss, custom_generator_loss
from utils import load_dataset

import PIL
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras import Input, Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Reshape, UpSampling2D, Conv2D, Activation, \
    concatenate, Flatten, Lambda, Concatenate
from keras.optimizers import Adam
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_model', type=str, default="/content/drive/My Drive/StackGAN/Data/Models/latest_model_face_temp.ckpt", help='Pre-trained model path to resume from')
    parser.add_argument('--save_every', type=int, default=60, help='Save Model/Samples every x iterations over batches')
    
    args = parser.parse_args()  
    data_dir = "/content/drive/My Drive/StackGAN/Data"
    train_dir = data_dir + "/train"
    test_dir = data_dir + "/test"
    image_size = 64
    batch_size = 32
    z_dim = 100
    stage1_generator_lr = 0.0002
    stage1_discriminator_lr = 0.0002
    stage1_lr_decay_step = 600
    epochs = 10
    condition_dim = 128

    embeddings_file_path_train = train_dir + "/training_caption_vectors.hdf5"
    embeddings_file_path_test = test_dir + "/testing_caption_vectors.hdf5"

    filenames_file_path_train = train_dir + "/training_images.txt"
    filenames_file_path_test = test_dir + "/testing_images.txt"

    #class_info_file_path_train = train_dir + "/class_info.pickle"
    #class_info_file_path_test = test_dir + "/class_info.pickle"

    #cub_dataset_dir = "/content/CUB_200_2011"
    
    # Define optimizers
    dis_optimizer = Adam(lr=stage1_discriminator_lr, beta_1=0.5, beta_2=0.999)
    gen_optimizer = Adam(lr=stage1_generator_lr, beta_1=0.5, beta_2=0.999)

    """"
    Load datasets
    """
    '''
    X_train, embeddings_train = load_dataset(images_path=filenames_file_path_train,
                                                      data_dir="/content/drive/My Drive/StackGAN/Data/face/jpg/",
                                                      skipthought_encodings_path=embeddings_file_path_train,
                                                      image_size=(64, 64),
                                                      start = 0,
                                                      end = 10000)

 
    X_test, embeddings_test = load_dataset(images_path=filenames_file_path_test,
                                                   data_dir="/content/drive/My Drive/StackGAN/Data/testing_images/",
                                                   skipthought_encodings_path=embeddings_file_path_test,
                                                   image_size=(64, 64),
                                                   start = 10000,
                                                   end = 12500)
    np.savez('loadDataset.npz', name1 = X_train, name2 = embeddings_train, name3 = X_test, name4 = embeddings_test)
    '''

    data = np.load("/content/drive/My Drive/StackGAN/loadDataset.npz")
    X_train = data['name1']
    embeddings_train = data['name2']
    X_test = data['name3']
    embeddings_test = data['name4']
    """
    Build and compile networks
    """
    ca_model = build_ca_model()
    ca_model.compile(loss="binary_crossentropy", optimizer="adam")

    stage1_dis = build_stage1_discriminator()
    stage1_dis.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    stage1_gen = build_stage1_generator()
    stage1_gen.compile(loss="mse", optimizer=gen_optimizer)

    embedding_compressor_model = build_embedding_compressor_model()
    embedding_compressor_model.compile(loss="binary_crossentropy", optimizer="adam")

    adversarial_model = build_adversarial_model(gen_model=stage1_gen, dis_model=stage1_dis)
    adversarial_model.compile(loss=['binary_crossentropy', KL_loss], loss_weights=[1, 2.0],
                              optimizer=gen_optimizer, metrics=None)

    if args.resume_model:
      adversarial_model.load_weights(args.resume_model)

    tensorboard = TensorBoard(log_dir="/content/drive/My Drive/StackGAN/Datalogs/".format(time.time()))
    tensorboard.set_model(stage1_gen)
    tensorboard.set_model(stage1_dis)
    tensorboard.set_model(ca_model)
    tensorboard.set_model(embedding_compressor_model)

    # Generate an array containing real and fake values
    # Apply label smoothing as well
    real_labels = np.ones((batch_size, 1), dtype=float) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=float) * 0.1

    for epoch in range(epochs):
        #print("========================================")
        print("Epoch is:", epoch)
        #print("Number of batches", int(X_train.shape[0] / batch_size))

        gen_losses = []
        dis_losses = []

        # Load data and train model
        number_of_batches = int(X_train.shape[0] / batch_size)
        for index in range(number_of_batches):
            #print("Batch:{}".format(index+1))
            
            """
            Train the discriminator network
            """
            # Sample a batch of data
            z_noise = np.random.normal(0, 1, size=(batch_size, z_dim))
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            embedding_batch = embeddings_train[index * batch_size:(index + 1) * batch_size]
            image_batch = (image_batch - 127.5) / 127.5

            # Generate fake images
            fake_images, _ = stage1_gen.predict([embedding_batch, z_noise], verbose=3)

            # Generate compressed embeddings
            compressed_embedding = embedding_compressor_model.predict_on_batch(embedding_batch)
            compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, condition_dim))
            compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))

            dis_loss_real = stage1_dis.train_on_batch([image_batch, compressed_embedding],
                                                      np.reshape(real_labels, (batch_size, 1)))
            dis_loss_fake = stage1_dis.train_on_batch([fake_images, compressed_embedding],
                                                      np.reshape(fake_labels, (batch_size, 1)))
            dis_loss_wrong = stage1_dis.train_on_batch([image_batch[:(batch_size - 1)], compressed_embedding[1:]],
                                                       np.reshape(fake_labels[1:], (batch_size-1, 1)))

            d_loss = 0.5 * np.add(dis_loss_real, 0.5 * np.add(dis_loss_wrong, dis_loss_fake))

            #print("d_loss_real:{}".format(dis_loss_real))
            #print("d_loss_fake:{}".format(dis_loss_fake))
            #print("d_loss_wrong:{}".format(dis_loss_wrong))
            #print("d_loss:{}".format(d_loss))

            """
            Train the generator network 
            """
            g_loss = adversarial_model.train_on_batch([embedding_batch, z_noise, compressed_embedding],[K.ones((batch_size, 1)) * 0.9, K.ones((batch_size, 256)) * 0.9])
            #print("g_loss:{}".format(g_loss))

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

            if(index % args.save_every) == 0:
              #print("Saving Model")
              adversarial_model.save_weights("/content/drive/My Drive/StackGAN/Data/Models/latest_model_face_temp.ckpt")

        """
        Save losses to Tensorboard after each epoch
        """
        write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        write_log(tensorboard, 'generator_loss', np.mean(gen_losses[0]), epoch)
        
        # Generate and save images after every 2nd epoch
        if epoch % 10 == 0:
            # z_noise2 = n+p.random.uniform(-1, 1, size=(batch_size, z_dim))
            z_noise2 = np.random.normal(0, 1, size=(batch_size, z_dim))
            embedding_batch = embeddings_test[0:batch_size]
            fake_images, _ = stage1_gen.predict_on_batch([embedding_batch, z_noise2])

            # Save images
            for i, img in enumerate(fake_images[:10]):
                save_rgb_img(img, "/content/drive/My Drive/StackGAN/Data/results/gen_{}_{}.png".format(epoch, i))

            # Save model
        if epoch % 20 == 0:
            adversarial_model.save_weights("/content/drive/My Drive/StackGAN/Data/Models/model_after_face_epoch_{}.ckpt".format(epoch))

    # Save models
    stage1_gen.save_weights("/content/drive/My Drive/StackGAN/Data/stage1_gen.h5")
    stage1_dis.save_weights("/content/drive/My Drive/StackGAN/Data/stage1_dis.h5")