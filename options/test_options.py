from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=3000, help='how many test images to run')
        parser.add_argument('--state', type=str, default='seenstyle_oov', help='in which scenario to test')
        parser.add_argument('--sty_refRoot', type=str, default='images/img_sty_reference.png', help='which style reference to test')
        parser.add_argument('--cont_refRoot', type=str, default='images/img_cont_reference.png', help='which content reference to test')
        parser.add_argument('--save_dir', type=str, default='images_iam', help='path to save reference img')
        parser.add_argument('--label', type=str,  help='which content to render')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
