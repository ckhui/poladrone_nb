from easydict import EasyDict

Cfg = EasyDict()
Cfg.batch = 2
Cfg.subdivisions = 2
Cfg.width = 416
Cfg.height = 416
Cfg.channels = 3
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0

Cfg.saturation = 1.1
Cfg.exposure = 1.1
Cfg.hue = .1

Cfg.learning_rate = 0.00261
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

Cfg.cutmix = 0
Cfg.mosaic = 1

Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 3
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0 ## got error with blur
Cfg.gaussian = 0
Cfg.boxes = 60  # box num
Cfg.TRAIN_EPOCHS = 300
Cfg.train_label = 'data/train.txt'
Cfg.val_label = 'data/val.txt'
Cfg.TRAIN_OPTIMIZER = 'adam'
'''
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

Cfg.checkpoints = 'checkpoints'
Cfg.TRAIN_TENSORBOARD_DIR = 'log'