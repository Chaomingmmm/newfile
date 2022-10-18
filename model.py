
# import packages
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D, Dense, Conv2DTranspose
from keras.layers import Dropout
from keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio



def build_generator():
    model = Sequential()
    model.add(Dense(256 * 8* 8, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8,8,256)))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2)) 
    # shape 32*32*128

    # model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
# shape 64*64*128
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))

    model.summary() 

    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3,3), padding='same', ))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model

def save_imgs(epoch,generator,img_channels):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, img_channels))
        gen_imgs = generator.predict(noise)
        global save_name
        save_name += 0.00000001

        # Rescale images 0 - 1
        gen_imgs = (gen_imgs + 1) / 2.0

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("currentgeneration.png")
        fig.savefig("generated_imgs/%.8f.png" % save_name)
        plt.close()



def train(epochs, batch_size=32, save_interval=200):

  array = []
  #PUT PATH OF RESIZED IMAGES
  path = "./resized_imgs"

  for dir in os.listdir(path):
            # print(dir)
    image = Image.open(os.path.join(path,dir))
    data = np.asarray(image)
    array.append(data)

  X_train = np.array(array)
  print(X_train.shape)

  # print(X_train.shape)
  #Rescale data between -1 and 1
  X_train = X_train / 127.5 -1.
  bat_per_epo = int(X_train.shape[0] / batch_size)
  # X_train = np.expand_dims(X_train, axis=3)
  print(X_train.shape)

  #Create our Y for our Neural Networks
  valid = np.ones((batch_size, 1))
  fakes = np.zeros((batch_size, 1))

  for epoch in range(epochs):
    for j in range(bat_per_epo):
      #Get Random Batch
      idx = np.random.randint(0, X_train.shape[0], batch_size)
      imgs = X_train[idx]

      #Generate Fake Images
      noise = np.random.normal(0, 1, (batch_size, latent_dim))
      gen_imgs = generator.predict(noise)

      #Train discriminator
      d_loss_real = discriminator.train_on_batch(imgs, valid)
      d_loss_fake = discriminator.train_on_batch(gen_imgs, fakes)
      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

      noise = np.random.normal(0, 1, (batch_size, latent_dim))
      
      #inverse y label
      g_loss = GAN.train_on_batch(noise, valid)

      print("******* %d %d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch,j, d_loss[0], 100* d_loss[1], g_loss))

      # if(epoch % save_interval) == 0:
    save_imgs(epoch,generator,latent_dim)
def save_GAN_weights(gw_path = './weights',dw_path = './weights'):
    generator.save_weights(gw_path + "generator.h5")
    discriminator.save_weights(dw_path + "discriminator.h5")

if __name__=="__main__":

    # set parameters for images
    img_width = 32
    img_height = 32
    channels = 3
    img_shape = (img_width, img_height, channels)

    # set parameters for model
    latent_dim = 100
    adam = Adam(lr=0.0002)

    # build model
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    GAN = Sequential()
    # discriminator.trainable = False
    GAN.add(generator)
    GAN.add(discriminator)

    GAN.compile(loss='binary_crossentropy', optimizer=adam)
    print('model built!')

    print('start training...')
    save_name = 0.00000000
    train(500, batch_size=32, save_interval=200)

    print('finished training!\n')

    # save weights
    gw_path = './weights'
    dw_path = './weights'
    save_GAN_weights(gw_path,dw_path)




