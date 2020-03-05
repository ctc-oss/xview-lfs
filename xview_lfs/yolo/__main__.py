from PIL import Image
from tqdm import tqdm
import logging
import configargparse as argparse
import os
import re
import xview_lfs as data
import xview.wv_util as wv
from . import write_yolo_labels, make_temp_dir
import lfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lfs_url", type=str, help="LFS repo URL")
    parser.add_argument("-r", "--ref", type=str, default='master', help="LFS data ref")
    parser.add_argument("-i", "--images", type=str, action='append', help="List of image ids to include; empty for all")
    parser.add_argument("-d", "--dictionary", type=str, help="Path to class dictionary; defaults to xview dict")
    parser.add_argument("-c", "--classes", type=str, help="Class ids from labels to include; empty for all")
    parser.add_argument("-k", "--insecure", action='store_true', help="Skip SSL verification; GIT_SSL_NO_VERIFY")
    parser.add_argument("-s", "--chip_size", type=int, default=544, help="Training chip size")
    parser.add_argument("-p", "--prune_empty", action='store_true', help='Prune empty chips')
    parser.add_argument("-w", "--workspace", help="Working directory")
    parser.add_argument("--chip_dir", type=str, default='chipped', help="Chip output dir (relative to workspace)")
    parser.add_argument("--chip_format", type=str, default='jpg', help="Training chip format (jpg, png, ...)")
    parser.add_argument("--yolo_root_dir", default='/opt/darknet', help="Yolo install dir")
    args = parser.parse_args()

    if args.insecure:
        os.environ['GIT_SSL_NO_VERIFY'] = '1'

    Image.init()
    if f'.{args.chip_format}' not in Image.EXTENSION.keys():
        raise SystemError(f'invalid chip format, {args.chip_format}')

    # prepare the working directory
    if not args.workspace:
        args.workspace = make_temp_dir('yolo-')
    chip_out_dir = os.path.join(args.workspace, args.chip_dir)
    if not os.path.exists(chip_out_dir):
        os.mkdir(chip_out_dir)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("yolo")

    boxes = {}
    skip_chips = 0
    images_list = []
    classes_actual = {}

    images = []
    if args.images:
        for img_list in args.images:
            images.extend(img_list.split(','))

    if images:
        logger.info('using images: %s' % images)

    logger.debug('------------ loading data --------------')
    res = (args.chip_size, args.chip_size)
    lfs_wd, d = data.load_train_data(images, url=args.lfs_url, ref=args.ref, chipsz=res)
    logger.info(f'lfs working directory: {lfs_wd}')

    class_dict = args.dictionary
    if not class_dict:
        default_labels = os.path.join(lfs_wd, 'labels.txt')
        if os.path.exists(default_labels):
            labels = wv.get_classes(default_labels)
        else:
            labels = wv.get_classes()
    elif class_dict == 'xview':
        labels = wv.get_classes()
    elif lfs.is_uri(class_dict):
        labels = wv.get_classes(lfs.get(class_dict))
    elif os.path.exists(class_dict):
        labels = wv.get_classes(class_dict)
    elif not class_dict.startswith('/') and os.path.exists(os.path.join(lfs_wd, class_dict)):
        labels = wv.get_classes(os.path.join(lfs_wd, class_dict))
    else:
        raise SystemError(f'invalid dictionary path, {class_dict}')

    logger.info(f'class dictionary: {class_dict}')

    excluded_classes = []
    if args.classes:
        splits = map(lambda s: int(s), args.classes.split(','))
        for rem in set(labels.keys()) - set(splits):
            excluded_classes.append(labels.pop(rem, None))

    tot_box = 0
    ind_chips = 0

    for iid in tqdm(d.keys()):
        im, box, classes_final = d[iid]

        for _, count in classes_final.items():
            for c in count:
                # todo;; cast class-id to int upstream
                classes_actual[int(c)] = classes_actual.get(c, 0) + 1

        for idx, image in enumerate(im):
            # todo;; cast class-id to int upstream
            int_classes_final = list(map(lambda cid: int(cid), classes_final[idx]))
            tf_example = write_yolo_labels(image, box[idx], int_classes_final, labels)

            if tf_example or not args.prune_empty:
                tot_box += tf_example.count('\n')

                chipid = str(ind_chips).rjust(6, '0')
                writer = open(os.path.join(chip_out_dir, f'{chipid}.txt'), "w")
                img_file = os.path.join(chip_out_dir, f'{chipid}.{args.chip_format}')
                Image.fromarray(image).save(img_file)
                images_list.append(img_file)

                writer.write(tf_example)
                writer.close()

                ind_chips += 1
            else:
                skip_chips += 1

    logger.info("Tot Box: %d" % tot_box)
    logger.info("Chips: %d" % ind_chips)
    logger.info("Skipped Chips: %d" % skip_chips)

    final_classes_map = []
    logger.info("Generating xview.pbtxt")
    with open(os.path.join(args.workspace, 'xview.pbtxt'), 'w') as f:
        filled_in_labels = data.fill_in_gaps_and_background(labels)
        for class_id, class_label in sorted(filled_in_labels.items()):
            name = filled_in_labels[class_id]
            f.write('\n'.join([
                "item {",
                f"  id: {class_id}",
                f"  name: '{name}'",
                "}\n"])
            )
            if class_id in classes_actual.keys():
                logger.info(' {:>3} {:25}{:>5}'.format(class_id, name, classes_actual[class_id]))
                final_classes_map.append(name)
        logger.debug(f"wrote %s" % f.name)

    logger.info("Generating training_list.txt")
    training_list_path = os.path.join(args.workspace, 'training_list.txt')
    with open(training_list_path, 'w') as f:
        f.write('\n'.join(images_list))
        f.write('\n')
        logger.debug("wrote %s" % f.name)

    yolo_cfg_src = os.path.join(args.yolo_root_dir, 'cfg', 'yolov3.cfg')
    if os.path.exists(yolo_cfg_src):
        logger.info("Darknet installation found, generating yolo configuration")

        logger.info("Generating obj.names")
        yolo_names_path = os.path.join(args.workspace, 'obj.names')
        with open(yolo_names_path, 'w') as f:
            labelstr = "\n".join(final_classes_map)
            f.write(labelstr)
            logger.debug("wrote %s" % f.name)

        logger.info("Generating obj.data")
        class_count = len(final_classes_map)
        yolo_obj_data_path = os.path.join(args.workspace, 'obj.data')
        with open(yolo_obj_data_path, 'w') as f:
            f.write(f'classes={class_count}\n')
            f.write(f'train={training_list_path}\n')
            f.write(f'valid={training_list_path}\n')  # todo;; separate val images
            f.write(f'names={yolo_names_path}\n')
            f.write(f'backup={os.path.join(args.workspace, "backup")}\n')
            logger.debug("wrote %s" % f.name)

        max_batches = max(4000, class_count * 2000)
        yolocfg = {
            'batch': '64',
            'subdivisions': '16',
            'width': f'{args.chip_size}',
            'height': f'{args.chip_size}',
            'max_batches': f'{max_batches}',
            'steps': f'{int(max_batches * .8)},{int(max_batches * .9)}',
            'filters': f'{(class_count + 5) * 3}',
            'classes': f'{class_count}'
        }

        logger.info("Generating yolo-obj.cfg")
        yolo_cfg_path = os.path.join(args.workspace, 'yolo-obj.cfg')
        with open(yolo_cfg_src, 'r') as source, open(yolo_cfg_path, 'w') as target:
            lines = source.readlines()
            for ln in lines:
                oln = ln
                for class_id, count in yolocfg.items():
                    if class_id == 'filters':  # todo;; hacked in here for special filters case
                        oln = re.sub(f'^{class_id}=255$', f'{class_id}={count}', oln)
                    else:
                        oln = re.sub(f'^{class_id}( )?=.+$', f'{class_id}={count}', oln)
                target.write(oln)

        logger.info(f'command:\t\tdarknet detector train {yolo_obj_data_path} {yolo_cfg_path} darknet53.conv.74')

    logger.info(f'file://{args.workspace}')
    print(args.workspace)
