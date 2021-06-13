""" pyTorch Demo

Usage:
    Main.py (-h | --help)
    Main.py --train [--ld=<pretrained_model>] --in=<input_source> --gt=<groundtruth_source> --inv=<valid_input_source> --gtv=<valid_groundtruth_source> [--cpu] --e=<max_epoch>
    Main.py --test --ld=<pretrained_model> --in=<test_input_source> [--out=<test_output>] [--cpu]

Options:
    -h --help                           Show this screen
    --train                             Train mode
    --test                              Test mode
    --ld=<pretrained_model>             Specify a model for loading
    --in=<input_source>                 Source for the input samples
    --gt=<groundtruth_source>           Source for the training groundtruth samples
    --inv=<valid_input_source>          Source for the validation input samples
    --gtv=<valid_groundtruth_source>    Source for the validation groundtruth samples
    --e=<max_epoch>                     Max. training epochs
    --cpu                               Force CPU mode
    --out=<test_output>                 Destination folder for the processed samples
"""



    

from docopt import docopt
from NetManager import NetManager

def main(arguments):
    net = NetManager()
    device = "cuda:0"
    if arguments['--cpu']:
        device = "cpu"
    net.set_device(device)

    if arguments['--ld'] is not None:
        net.load(arguments['--ld'])
    if arguments['--train'] and (arguments['--in'] is not None) and (arguments['--gt'] is not None)  and (arguments['--inv'] is not None) and (arguments['--gtv'] is not None) and (arguments['--e'] is not None):
        net.train(arguments['--in'], arguments['--gt'], arguments['--inv'], arguments['--gtv'] ,int(arguments['--e']))
    elif arguments['--test'] and (arguments['--in'] is not None) and (arguments['--out'] is not None):
        net.test(arguments['--in'], arguments['--out'])
    else:
        print(arguments)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
