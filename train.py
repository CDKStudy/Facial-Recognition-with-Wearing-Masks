import os
from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.siamese import siamese
from nets.siamese_training import Generator
from utils.callbacks import ModelCheckpoint
from utils.utils_fit import fit_one_epoch
import matplotlib.pyplot as plt

#----------------------------------------------------#
#   计算图片总数
#----------------------------------------------------#
def get_image_num(path, train_own_data):
    num = 0
    if train_own_data:
        train_path = os.path.join(path, 'images_background')
        for character in os.listdir(train_path):
            #----------------------------------------------------#
            #   在大众类下遍历小种类。
            #----------------------------------------------------#
            character_path = os.path.join(train_path, character)
            num += len(os.listdir(character_path))
    else:
        train_path = os.path.join(path, 'images_background')
        for alphabet in os.listdir(train_path):
            #-------------------------------------------------------------#
            #   然后遍历images_background下的每一个文件夹，代表一个大种类
            #-------------------------------------------------------------#
            alphabet_path = os.path.join(train_path, alphabet)
            for character in os.listdir(alphabet_path):
                #----------------------------------------------------#
                #   在大众类下遍历小种类。
                #----------------------------------------------------#
                character_path = os.path.join(alphabet_path, character)
                num += len(os.listdir(character_path))
    return num
  
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    #------------------------------------------------------#
    #   是否使用eager模式训练
    #------------------------------------------------------#
    eager           = False
    #----------------------------------------------------#
    #   数据集存放的路径
    #----------------------------------------------------#
    dataset_path    = "datasets"
    #----------------------------------------------------#
    #   训练好的权值保存在logs文件夹里面
    #----------------------------------------------------#
    log_dir         = "logs/"
    #----------------------------------------------------#
    #   输入图像的大小，默认为105,105,3
    #----------------------------------------------------#
    input_shape     = [32,32,3]
    #----------------------------------------------------#
    #   当训练Omniglot数据集时设置为False
    #   当训练自己的数据集时设置为True
    #
    #   训练自己的数据和Omniglot数据格式不一样。
    #   详情可看README.md
    #----------------------------------------------------#
    train_own_data  = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的训练参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，此时从0开始训练。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    model = siamese(input_shape)
    if model_path != '':
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
    
    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    tensorboard         = TensorBoard(log_dir=log_dir)
    checkpoint_period   = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr           = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping      = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    #----------------------------------------------------#
    #   训练集和验证集的比例。
    #----------------------------------------------------#
    train_ratio         = 0.8
    images_num          = get_image_num(dataset_path, train_own_data)
    num_train           = int(images_num * train_ratio)
    num_val             = images_num - num_train
    
    # -------------------------------------------------------------#
    #   训练分为两个阶段，两阶段初始的学习率不同，手动调节了学习率
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    # -------------------------------------------------------------#
    if True:
        Batch_size      = 16#32
        Lr              = 1e-4
        Init_epoch      = 0
        Freeze_epoch    = 40#50

        epoch_step          = num_train // Batch_size
        epoch_step_val      = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        generator = Generator(input_shape, dataset_path, Batch_size, train_ratio, train_own_data)

        print('Train with batch size {}.'.format(Batch_size))
        if eager:
            gen = tf.data.Dataset.from_generator(partial(generator.generate, train = True), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(generator.generate, train = False), (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = Lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

            for epoch in range(Init_epoch, Freeze_epoch):
                fit_one_epoch(model, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Freeze_epoch)
        else:
            model.compile(loss = "binary_crossentropy", optimizer = Adam(lr=Lr), metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC()])
            model.summary()

            history=model.fit_generator(
                generator.generate(True),
                steps_per_epoch     = epoch_step,
                validation_data     = generator.generate(False),
                validation_steps    = epoch_step_val,
                epochs              = Freeze_epoch,
                initial_epoch       = Init_epoch,
                callbacks           = [checkpoint_period, reduce_lr, early_stopping, tensorboard]
            )
            plt.plot(history.history['binary_accuracy'], 'bo', label='Training binary_acc')
            plt.plot(history.history['val_binary_accuracy'], 'b', label='Validation binary_acc')
            plt.title('Training and validation binary_accuracy')
            plt.legend()
            plt.figure()

            plt.plot(history.history['loss'], 'bo', label='Training loss')
            plt.plot(history.history['val_loss'], 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.figure()
            plt.show()
            model.save('model')


    if True:
        Batch_size      = 16#32
        Lr              = 1e-5
        Freeze_epoch    =40 #50
        Epoch           =80#100
            
        epoch_step          = num_train // Batch_size
        epoch_step_val      = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        generator = Generator(input_shape, dataset_path, Batch_size, train_ratio, train_own_data)

        print('Train with batch size {}.'.format(Batch_size))
        if eager:
            gen = tf.data.Dataset.from_generator(partial(generator.generate, train = True), (tf.float32, tf.float32))
                
            gen_val = tf.data.Dataset.from_generator(partial(generator.generate, train = False), (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = Lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

            for epoch in range(Freeze_epoch, Epoch):
                fit_one_epoch(model, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch)
        else:
            model.compile(loss = "binary_crossentropy", optimizer = Adam(lr=Lr), metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC()])

            history=model.fit_generator(
                generator.generate(True),
                steps_per_epoch     = epoch_step,
                validation_data     = generator.generate(False),
                validation_steps    = epoch_step_val,
                epochs              = Epoch,
                initial_epoch       = Freeze_epoch,
                callbacks           = [checkpoint_period, reduce_lr, early_stopping, tensorboard]
            )
            plt.plot(history.history['binary_accuracy'],'bo', label='Training binary_acc')
            plt.plot(history.history['val_binary_accuracy'],'b', label='Validation binary_acc')
            plt.title('Training and validation binary_accuracy')
            plt.legend()
            plt.figure()

            plt.plot(history.history['loss'],'bo', label='Training loss')
            plt.plot(history.history['val_loss'],'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.figure()
            plt.show()
            model.save('model')
