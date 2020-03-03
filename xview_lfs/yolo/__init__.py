import os
import tempfile


def write_yolo_labels(img, boxes, class_num, labels):
    sw = img.shape[0]
    sh = img.shape[1]

    yolo_text = []
    for ind, box in enumerate(boxes):
        if not class_num[ind] in labels:
            continue

        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])

        if xmin + ymin + xmax + ymax > 0:
            dw = 1. / sw
            dh = 1. / sh
            xmid = (xmin + xmax) / 2.0
            ymid = (ymin + ymax) / 2.0
            w0 = xmax - xmin
            h0 = ymax - ymin
            x = xmid * dw
            y = ymid * dh
            w = w0 * dw
            h = h0 * dh

            yolo_text.append(f"{class_num[ind]} {x} {y} {w} {h}")

    return '\n'.join(yolo_text)


def make_temp_dir(pre):
    td = tempfile.mkdtemp(prefix=pre)
    os.chmod(td, 0o755)
    return td
