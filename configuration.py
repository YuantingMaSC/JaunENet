DEVICE = "gpu"   # cpu or gpu
import tensorflow as tf
# some training parameters

init_lr = 3e-5
EPOCHS = 500
save_every_n_epoch = 25
BATCH_SIZE = 128
NUM_CLASSES = 3
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3

metasize = 9


dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"


