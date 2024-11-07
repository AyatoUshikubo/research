import numpy as np
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv3D, Conv2D
from keras.layers import Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers import Reshape, Input
from keras.layers import GlobalAveragePooling2D, Multiply
from keras.layers import BatchNormalization
from keras.utils import plot_model
from keras.layers import Add, Subtract, Multiply, Concatenate
from keras import layers
from scipy import stats

# Architectures
def residual(inputs, n_filters):
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, inputs])
    return x
# Networks
def UNet(inputs, n_filters, kernel_size, n_outputs):
    
    conv1 = Conv2D(n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.5)(conv1)
    down1 = MaxPooling2D((2,2))(conv1)

    conv2 = Conv2D(2*n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(down1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2*n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.5)(conv2)
    down2 = MaxPooling2D((2,2))(conv2)
    
#     conv3 = Conv2D(4*n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(down2)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Activation('relu')(conv3)
#     conv3 = Conv2D(4*n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(conv3)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Activation('relu')(conv3)
#     conv3 = Dropout(0.5)(conv3)
#     down3 = MaxPooling2D((2,2))(conv3)

    convc = Conv2D(4 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(down2)
    convc = BatchNormalization()(convc)
    convc = Activation('relu')(convc)
    convc = Conv2D(4 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(convc)
    convc = BatchNormalization()(convc)
    convc = Activation('relu')(convc)
    convc = Dropout(0.5)(convc)
    
#     convc = Conv2D(8 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(down3)
#     convc = BatchNormalization()(convc)
#     convc = Activation('relu')(convc)
#     convc = Conv2D(8 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(convc)
#     convc = BatchNormalization()(convc)
#     convc = Activation('relu')(convc)
#     convc = Dropout(0.5)(convc)
    
#     upconv3 = Concatenate(axis=3)([conv3, UpSampling2D(size=(2, 2))(convc)])
#     upconv3 = Conv2D(4 * n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(upconv3)
#     upconv3 = BatchNormalization()(upconv3)
#     upconv3 = Activation('relu')(upconv3)
#     upconv3 = Conv2D(4 * n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(upconv3)
#     upconv3 = BatchNormalization()(upconv3)
#     upconv3 = Activation('relu')(upconv3)
#     upconv3 = Dropout(0.5)(upconv3)

    upconv2 = Concatenate(axis=3)([conv2, UpSampling2D(size=(2, 2))(convc)])
#     upconv2 = Concatenate(axis=3)([conv2, UpSampling2D(size=(2, 2))(upconv3)])
    upconv2 = Conv2D(2 * n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(upconv2)
    upconv2 = BatchNormalization()(upconv2)
    upconv2 = Activation('relu')(upconv2)
    upconv2 = Conv2D(2 * n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(upconv2)
    upconv2 = BatchNormalization()(upconv2)
    upconv2 = Activation('relu')(upconv2)
    upconv2 = Dropout(0.5)(upconv2)

    upconv1 = Concatenate(axis=3)([conv1, UpSampling2D(size=(2, 2))(upconv2)])
    upconv1 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(upconv1)
    upconv1 = BatchNormalization()(upconv1)
    upconv1 = Activation('relu')(upconv1)
    upconv1 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(upconv1)
    upconv1 = BatchNormalization()(upconv1)
    upconv1 = Activation('relu')(upconv1)
    upconv1 = Dropout(0.5)(upconv1)

    outputs = Conv2D(n_outputs, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='linear')(upconv1)

    return Model(inputs=inputs, outputs=outputs)

def UNet3D(inputs, n_filters, kernel_size, n_outputs):
    conv1 = Conv3D(n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu')(inputs)

def DeepVel(inputs, n_filters, n_res):
    conv = Conv2D(n_filters, (3,3), padding='same', kernel_initializer='he_normal')(inputs)
    x = residual(conv, n_filters)
    for i in range(n_res-1):
        x = residual(x, n_filters)
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, conv])
    outputs = Conv2D(1, (1, 1), activation='linear', padding='same', kernel_initializer='he_normal')(x)
    return Model(inputs=inputs, outputs=outputs)

def DeepVelU(input_dim, n_filters, kernel_size, n_outputs, activation, activation_output):

    inputs = Input(shape=input_dim)

    conv1 = Conv2D(n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)
    conv1 = Dropout(0.5)(conv1)
    stri1 = Conv2D(n_filters, kernel_size, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
    stri1 = BatchNormalization()(stri1)
    stri1 = Activation(activation)(stri1)

    conv2 = Conv2D(2 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(stri1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)
    conv2 = Dropout(0.5)(conv2)
    stri2 = Conv2D(2 * n_filters, kernel_size, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
    stri2 = BatchNormalization()(stri2)
    stri2 = Activation(activation)(stri2)

    conv3 = Conv2D(2 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(stri2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)
    conv3 = Dropout(0.5)(conv3)
    stri3 = Conv2D(2 * n_filters, kernel_size, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
    stri3 = BatchNormalization()(stri3)
    stri3 = Activation(activation)(stri3)

    convc = Conv2D(4 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(stri3)
    convc = BatchNormalization()(convc)
    convc = Activation(activation)(convc)
    convc = Conv2D(4 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(convc)
    convc = BatchNormalization()(convc)
    convc = Activation(activation)(convc)
    convc = Dropout(0.5)(convc)

    upconv3 = Conv2D(2 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu')(UpSampling2D(size=(2, 2))(convc))
    upconv3 = Concatenate(axis=3)([conv3, upconv3])
    upconv3 = Conv2D(2 * n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(upconv3)
    upconv3 = BatchNormalization()(upconv3)
    upconv3 = Activation(activation)(upconv3)
    upconv3 = Conv2D(2 * n_filters, kernel_size, strides=(1, 1), padding='same',
                     kernel_initializer='he_normal')(upconv3)
    upconv3 = BatchNormalization()(upconv3)
    upconv3 = Activation(activation)(upconv3)
    upconv3 = Dropout(0.5)(upconv3)

    upconv2 = Conv2D(2 * n_filters, kernel_size, strides=(1, 1), padding='same',
                     kernel_initializer='he_normal', activation='relu')(UpSampling2D(size=(2, 2))(upconv3))
    upconv2 = Concatenate(axis=3)([conv2, upconv2])
    upconv2 = Conv2D(2 * n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(upconv2)
    upconv2 = BatchNormalization()(upconv2)
    upconv2 = Activation(activation)(upconv2)
    upconv2 = Conv2D(2 * n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(upconv2)
    upconv2 = BatchNormalization()(upconv2)
    upconv2 = Activation(activation)(upconv2)
    upconv2 = Dropout(0.5)(upconv2)

    upconv1 = Conv2D(n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu')(UpSampling2D(size=(2, 2))(upconv2))
    upconv1 = Concatenate(axis=3)([conv1, upconv1])
    upconv1 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(upconv1)
    upconv1 = BatchNormalization()(upconv1)
    upconv1 = Activation(activation)(upconv1)
    upconv1 = Conv2D(n_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal')(upconv1)
    upconv1 = BatchNormalization()(upconv1)
    upconv1 = Activation(activation)(upconv1)
    upconv1 = Dropout(0.5)(upconv1)

    outputs = Conv2D(n_outputs, (1, 1), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal', activation=activation_output)(upconv1)
    
#     outputs = Conv2D(n_outputs, (1, 1), strides=(1, 1), padding='same', activation='linear')(outputs)

    return Model(inputs=inputs, outputs=outputs, )

def full_connect(input_dim, layer_sizes, output_dim, activation, activation_output):
    """
    指定されたレイヤーサイズと出力次元を持つ多層パーセプトロン（MLP）を作成します。
    レイヤーサイズに0から1の小数が含まれる場合、その数をドロップアウト率としてドロップアウト層を追加します。
    
    パラメータ:
    input_dim (int): 入力特徴量の次元。
    layer_sizes (list of int/float): 各要素がレイヤーのニューロン数またはドロップアウト率を表すリスト。
    output_dim (int): 出力の次元。
    
    戻り値:
    model (tf.keras.Model): 構築されたMLPモデル。
    """
    inputs = Input(shape=(input_dim,))
    
    x = inputs
    for size in layer_sizes:
        if isinstance(size, float) and 0 <= size <= 1:
            x = Dropout(rate=size)(x)
        else:
            x = Dense(size, activation=activation)(x)
    
    outputs = Dense(output_dim, activation=activation_output)(x)
    
    return Model(inputs=inputs, outputs=outputs)

def deeplab(input_dim, n_outputs):

    Nx = 128
    Ny = 128
    Nf1 = 60
    Nc1 = 3
    Nf2 = 40
    Nc2 = 7
    Nf3 = 20
    Nc3 = 15
    Nf4 = 10
    Nc4 = 31
    Nf5 = 5
    Nc5 = 51

    inputs = Input(shape=(input_dim))

    x1 = Conv2D(Nf1, (Nc1,Nc1), padding='same', kernel_initializer='he_normal')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x12 = Conv2D(Nf2, (Nc2,Nc2), padding='same',kernel_initializer='he_normal')(inputs)
    x12 = BatchNormalization()(x12)
    x12 = Activation('relu')(x12)
    x13 = Conv2D(Nf3, (Nc3,Nc3), padding='same',kernel_initializer='he_normal')(inputs)
    x13 = BatchNormalization()(x13)
    x13 = Activation('relu')(x13)
    x14 = Conv2D(Nf4, (Nc4,Nc4), padding='same',kernel_initializer='he_normal')(inputs)
    x14 = BatchNormalization()(x14)
    x14 = Activation('relu')(x14)
    x15 = Conv2D(Nf5, (Nc5,Nc5), padding='same',kernel_initializer='he_normal')(inputs)
    x15 = BatchNormalization()(x15)
    x15 = Activation('relu')(x15)
    x3 = keras.layers.concatenate([inputs,x1,x12,x13,x14,x15],axis=-1)
    x0 = Reshape((Nx,Ny,-1))(x3)

    ## SE block
    xse = GlobalAveragePooling2D()(x0)
    xse = Dense(30, activation='relu')(xse)
    xse = Dense(Nf1+Nf2+Nf3+Nf4+Nf5+input_dim[2], activation='sigmoid')(xse)
    x0 = Multiply()([x0,xse])
    
    ## Second layer
    x71 = Conv2D(20, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x71 = BatchNormalization()(x71)
    x72 = Conv2D(10, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x72 = BatchNormalization()(x72)
    x73 = Conv2D(5, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x73 = BatchNormalization()(x73)
    x74 = Conv2D(5, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x74 = BatchNormalization()(x74)
    x75 = Conv2D(2, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x75 = BatchNormalization()(x75)
    x1 = Conv2D(Nf1, (Nc1,Nc1), padding='same',kernel_initializer='he_normal')(x71)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x12 = Conv2D(Nf2, (Nc2,Nc2), padding='same',kernel_initializer='he_normal')(x72)
    x12 = BatchNormalization()(x12)
    x12 = Activation('relu')(x12)
    x13 = Conv2D(Nf3, (Nc3,Nc3), padding='same',kernel_initializer='he_normal')(x73)
    x13 = BatchNormalization()(x13)
    x13 = Activation('relu')(x13)
    x14 = Conv2D(Nf4, (Nc4,Nc4), padding='same',kernel_initializer='he_normal')(x74)
    x14 = BatchNormalization()(x14)
    x14 = Activation('relu')(x14)
    x15 = Conv2D(Nf5, (Nc5,Nc5), padding='same',kernel_initializer='he_normal')(x75)
    x15 = BatchNormalization()(x15)
    x15 = Activation('relu')(x15)
    x0 = keras.layers.concatenate([x0,x1,x12,x13,x14,x15],axis=-1)

    ## Output
    # main_output = Conv2D(n_outputs,(1,1), activation='linear',kernel_initializer='he_normal')(x0)
    main_output = Conv2D(n_outputs,(1,1), activation='relu',kernel_initializer='he_normal')(x0)
    return Model(inputs=inputs,outputs=main_output)

def deeplab_3D(input_dim, n_outputs):

    dh = 3
    Nx = 128
    Ny = 128
    Nf1 = 60
    Nc1 = 3
    Nf2 = 40
    Nc2 = 7
    Nf3 = 20
    Nc3 = 15
    Nf4 = 10
    Nc4 = 31
    Nf5 = 5
    Nc5 = 51

    inputs = Input(shape=(input_dim))

    x1 = Conv3D(Nf1, (Nc1,Nc1,dh), padding='same', kernel_initializer='he_normal')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x12 = Conv3D(Nf2, (Nc2,Nc2,dh), padding='same',kernel_initializer='he_normal')(inputs)
    x12 = BatchNormalization()(x12)
    x12 = Activation('relu')(x12)
    x13 = Conv3D(Nf3, (Nc3,Nc3,dh), padding='same',kernel_initializer='he_normal')(inputs)
    x13 = BatchNormalization()(x13)
    x13 = Activation('relu')(x13)
    x14 = Conv3D(Nf4, (Nc4,Nc4,dh), padding='same',kernel_initializer='he_normal')(inputs)
    x14 = BatchNormalization()(x14)
    x14 = Activation('relu')(x14)
    x15 = Conv3D(Nf5, (Nc5,Nc5,dh), padding='same',kernel_initializer='he_normal')(inputs)
    x15 = BatchNormalization()(x15)
    x15 = Activation('relu')(x15)
    x3 = keras.layers.concatenate([inputs,x1,x12,x13,x14,x15],axis=-1)
    x0 = Reshape((Nx,Ny,-1))(x3)

    ## SE block
    xse = GlobalAveragePooling2D()(x0)
    xse = Dense(30, activation='relu')(xse)
    # xse = Dense(Nf1+Nf2+Nf3+Nf4+Nf5+input_dim[3], activation='sigmoid')(xse)
    xse = Dense(414, activation='sigmoid')(xse)
    x0 = Multiply()([x0,xse])
    
    ## Second layer
    x71 = Conv2D(20, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x71 = BatchNormalization()(x71)
    x72 = Conv2D(10, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x72 = BatchNormalization()(x72)
    x73 = Conv2D(5, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x73 = BatchNormalization()(x73)
    x74 = Conv2D(5, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x74 = BatchNormalization()(x74)
    x75 = Conv2D(2, (1,1), padding='same',kernel_initializer='he_normal')(x0)
    x75 = BatchNormalization()(x75)
    x1 = Conv2D(Nf1, (Nc1,Nc1), padding='same',kernel_initializer='he_normal')(x71)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x12 = Conv2D(Nf2, (Nc2,Nc2), padding='same',kernel_initializer='he_normal')(x72)
    x12 = BatchNormalization()(x12)
    x12 = Activation('relu')(x12)
    x13 = Conv2D(Nf3, (Nc3,Nc3), padding='same',kernel_initializer='he_normal')(x73)
    x13 = BatchNormalization()(x13)
    x13 = Activation('relu')(x13)
    x14 = Conv2D(Nf4, (Nc4,Nc4), padding='same',kernel_initializer='he_normal')(x74)
    x14 = BatchNormalization()(x14)
    x14 = Activation('relu')(x14)
    x15 = Conv2D(Nf5, (Nc5,Nc5), padding='same',kernel_initializer='he_normal')(x75)
    x15 = BatchNormalization()(x15)
    x15 = Activation('relu')(x15)
    x0 = keras.layers.concatenate([x0,x1,x12,x13,x14,x15],axis=-1)

    ## Output
    main_output = Conv2D(n_outputs,(1,1), activation='linear',kernel_initializer='he_normal')(x0)
    return Model(inputs=inputs,outputs=main_output)








