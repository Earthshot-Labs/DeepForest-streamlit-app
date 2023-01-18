import argparse
from utils import configure


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--debug', action='store_true', help='for checking code')

        self.parser.add_argument('--batch_size', type=int, default=1, help='the number of batch_size')
        self.parser.add_argument('--dataset_name', type=str, default='Pix2PixHD', help='[Pix2PixHD]')
        self.parser.add_argument('--dataset_dir', type=str, default='images/content-images/', help='[path to dataset]')
        self.parser.add_argument('--gpu_ids', type=int, default=0, help='gpu number. If -1, use cpu')
        self.parser.add_argument('--input_ch', type=int, default=4, help='input channels')
        self.parser.add_argument('--output_ch', type=int, default=3, help='output channels')
        self.parser.add_argument('--image_height', type=int, default=1024, help='[512, 1024]')
        self.parser.add_argument('--n_downsample', type=int, default=4,
                                 help='how many times you want to downsample input data in G')
        self.parser.add_argument('--n_residual', type=int, default=9, help='the number of residual blocks in G')
        self.parser.add_argument('--n_workers', type=int, default=4, help='how many threads you want to use')
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d',
                                 help='[BatchNorm2d, InstanceNorm2d]')
        self.parser.add_argument('--padding_type', type=str, default='reflection',
                                 help='[reflection, replication, zero]')
        self.parser.add_argument('--use_boundary_map', action='store_true', default=True,
                                 help='if you want to use boundary map')

    def parse(self):
        self.opt = self.parser.parse_args()
        configure(self.opt)

        return self.opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', default=True, help='train flag')

        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='when to start decay the lr')
        self.parser.add_argument('--epoch_save', type=int, default=10)
        self.parser.add_argument('--eps', type=float, default=1e-8)
        self.parser.add_argument('--FM', action='store_true', default=True, help='switch for feature matching loss')
        self.parser.add_argument('--flip', action='store_true', default=True, help='switch for flip input data')
        self.parser.add_argument('--GAN_type', type=str, default='LSGAN', help='[GAN, LSGAN, WGAN_GP]')
        self.parser.add_argument('--lambda_FM', type=int, default=10, help='weight for FM loss')
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--iter_display', type=int, default=500)
        self.parser.add_argument('--iter_report', type=int, default=5)
        self.parser.add_argument('--shuffle', action='store_true', default=True,
                                 help='if you want to shuffle the order')
        self.parser.add_argument('--n_D', type=int, default=2,
                                 help='how many discriminators in different scales you want to use')
        self.parser.add_argument('--n_epochs', type=int, default=200, help='how many epochs you want to train')
        self.parser.add_argument('--VGG_loss', action='store_true',
                                 help='if you want to use VGGNet for additional feature matching loss')
        self.parser.add_argument('--is_inference', action='store_true', help='inference flag')


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', help='test flag')
        self.parser.add_argument('--is_inference', action='store_true', help='inference flag')

        self.parser.add_argument('--shuffle', action='store_true',
                                 help='if you want to shuffle the order')


class InferenceOption(BaseOption):
    def __init__(self):
        super(InferenceOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', help='test flag')
        self.parser.add_argument('--is_inference', action='store_true', default=True, help='inference flag')
        self.parser.add_argument('--checkpoint_epoch', type=int, default=300, help='number of the epoch to use for inference')
        self.parser.add_argument('--shuffle', action='store_true', default=False,
                                 help='if you want to shuffle the order')

