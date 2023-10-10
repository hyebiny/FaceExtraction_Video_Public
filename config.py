from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})
# to indicate this is a default setting, should not be changed by user
CONFIG.is_default = True
CONFIG.version = "baseline"
CONFIG.phase = "train"
# distributed training
CONFIG.dist = True
# global variables which will be assigned in the runtime
CONFIG.local_rank = 0
CONFIG.gpu = 0
CONFIG.world_size = 1

# Model config
CONFIG.model = EasyDict({})
# use pretrained checkpoint as encoder
# CONFIG.model.imagenet_pretrain = True
# CONFIG.model.imagenet_pretrain_path = "/home/liyaoyi/Source/python/attentionMatting/pretrain/model_best_resnet34_En_nomixup.pth"
CONFIG.model.batch_size = 16
CONFIG.model.mask_channel = 0
CONFIG.model.model_variant = 'mobilenetv3' # resnet50

# Training config
CONFIG.train = EasyDict({})
CONFIG.train.val_epoch = 1
# basic learning rate of optimizer
CONFIG.train.lr_backbone = 1e-4
CONFIG.train.lr_aspp = 2e-4
CONFIG.train.lr_decoder = 2e-4
CONFIG.train.lr_refiner = 0
# beta1 and beta2 for Adam
CONFIG.train.beta1 = 0.5
CONFIG.train.beta2 = 0.999
# weight of different losses
CONFIG.train.rec_weight = 1
CONFIG.train.lap_weight = 1
CONFIG.train.com_weight = 1
# clip large gradient
CONFIG.train.clip_grad = True
# ckpt
CONFIG.train.ckpt = None # "checkpoint/stage1/epoch-19.pth"
CONFIG.train.ckpt_dir = "checkpoint/stage1"
CONFIG.train.epoch_start = 0
CONFIG.train.epoch_end = 20
CONFIG.train.train_hr = False


# Dataloader config
CONFIG.data = EasyDict({})
CONFIG.data.name = "occmatte" # "videomatte", "imagematte"
CONFIG.data.workers = 0
CONFIG.data.trimap = False
# data path for training and validation in training phase
CONFIG.data.occ_root = None
CONFIG.data.fg_dir = None
CONFIG.data.bg_txt = None
CONFIG.data.rand_dir = None
CONFIG.data.folder_list = None
CONFIG.data.sim_list1 = None
CONFIG.data.sim_list2 = None
CONFIG.data.num_sample = 2
CONFIG.data.occlusion = 0.25
# stage 1 2 3 
CONFIG.data.resolution_lr = 512
CONFIG.data.resolution_hr = 2048
CONFIG.data.seq_length_lr = 4
CONFIG.data.seq_length_hr = 4
CONFIG.data.varT = 15


# Logging config
CONFIG.log = EasyDict({})
CONFIG.log.experiment_root = "./logs/stage1"
CONFIG.log.tensorboard_path = "tensorboard"
CONFIG.log.tensorboard_step = 100
# save less images to save disk space
CONFIG.log.tensorboard_image_step = 500
CONFIG.log.logging_path = "stout"
CONFIG.log.logging_step = 10
CONFIG.log.logging_level = "DEBUG"
CONFIG.log.checkpoint_path = "checkpoints"
CONFIG.log.checkpoint_step = 10000
CONFIG.log.disable_mixed_precision = False
CONFIG.log.disable_progress_bar = False



def load_config(custom_config, default_config=CONFIG, prefix="CONFIG"):
    """
    This function will recursively overwrite the default config by a custom config
    :param default_config:
    :param custom_config: parsed from config/config.toml
    :param prefix: prefix for config key
    :return: None
    """
    if "is_default" in default_config:
        default_config.is_default = False

    for key in custom_config.keys():
        full_key = ".".join([prefix, key])
        if key not in default_config:
            raise NotImplementedError("Unknown config key: {}".format(full_key))
        elif isinstance(custom_config[key], dict):
            if isinstance(default_config[key], dict):
                load_config(default_config=default_config[key],
                            custom_config=custom_config[key],
                            prefix=full_key)
            else:
                raise ValueError("{}: Expected {}, got dict instead.".format(full_key, type(custom_config[key])))
        else:
            if isinstance(default_config[key], dict):
                raise ValueError("{}: Expected dict, got {} instead.".format(full_key, type(custom_config[key])))
            else:
                default_config[key] = custom_config[key]