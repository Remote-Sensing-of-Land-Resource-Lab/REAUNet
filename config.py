import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='training device')
    parser.add_argument('--in_channels', type=int, default=3, help='channel number of the input samples')
    parser.add_argument('--num_class', type=int, default=1, help='the number of model output(s)')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='multiples of change in learning rate')
    parser.add_argument('--lr_step', type=int, default=30,
                        help='learning rate times {lr_gamma} after {lr_step} epochs when the loss no longer decreases')
    parser.add_argument('--max_epochs', type=int, default=200, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for determining extracted area')
    parser.add_argument('--checkpoint_folder', type=str, default=r'.\checkpoint', help='folder to save models')
    parser.add_argument('--save_epoch', type=int, default=20, help='save model after {save_epoch} epochs')
    parser.add_argument('--early_stop', type=int, default=50, help='early stop when loss stops decreasing')

    # required args
    parser.add_argument('--txt_path_train', type=str, required=True, help='path to the training text file')
    parser.add_argument('--txt_path_val', type=str, required=True, help='path to the valid text file')
    parser.add_argument('--image_root', type=str, required=True, help='file folder of the training samples')
    parser.add_argument('--label_root', type=str, required=True, help='file folder of the labels')

    # show args or not
    parser.add_argument('--output', default=True, help='show all arguments')

    args = parser.parse_args()

    if args.output:
        print('=' * 20, 'Args', '=' * 20)
        for arg, value in vars(args).items():
            print(f'{arg} -- {value}')

    return args


if __name__ == '__main__':
    args = get_args()