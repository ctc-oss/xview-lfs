import os

import configargparse as argparse

from . import load_train_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lfs_url", type=str, help="LFS repo URL")
    parser.add_argument("-r", "--ref", type=str, default='master', help="LFS data ref")
    parser.add_argument("-i", "--images", type=str, action='append', help="List of image ids to include; empty for all")
    parser.add_argument("-k", "--insecure", action='store_true', help="Skip SSL verification; GIT_SSL_NO_VERIFY")
    parser.add_argument("-s", "--chip_size", type=int, default=544, help="Training chip size")
    args = parser.parse_args()

    if args.insecure:
        os.environ['GIT_SSL_NO_VERIFY'] = '1'

    images = []
    if args.images:
        for img_list in args.images:
            images.extend(img_list.split(','))

    wd, _ = load_train_data(images, args.lfs_url, ref=args.ref, chipsz=(args.chip_size, args.chip_size))

    print(wd)
