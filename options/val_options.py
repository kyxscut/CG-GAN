from .base_options import BaseOptions


class ValOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results_val/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=3000, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='val')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        #radical_ateention parameters
        """
        parser.add_argument('--class_K', type=int, default=365, help='total numbers of radicals')
        parser.add_argument('--n', type=int, default=256, help='frequency of saving the latest results')
        parser.add_argument('--m', type=int, default=256, help='frequency of saving the latest results')
        parser.add_argument('--M', type=int, default=512, help='frequency of saving the latest results')
        parser.add_argument('--dim_attention', type=int, default=512, help='frequency of saving the latest results')
        parser.add_argument('--D', type=int, default=256, help='frequency of saving the latest results')
        """
        parser.add_argument('--hidden_size',type=int, default=256)
        parser.add_argument('--dropout_p',type=float,default =0.1 )
        parser.add_argument('--max_length',type = int,default = 64)
        #parser.add_argument('--dictionaryRoot', type=str, required=True, help='path to radical_dictionary')
        parser.add_argument('--dictionaryRoot', type=str, required=True, help='path to radical_dictionary')
        self.isTrain = False
        return parser
