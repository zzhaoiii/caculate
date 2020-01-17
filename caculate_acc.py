import xml.etree.ElementTree as ET
import os
import shutil
import cv2
import json

CLASSES = []


def convert(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall('object')
    gt = []

    for element in objects:
        org_name = element.find('name').text
        # name = str(class_to_ind[org_name])
        if org_name not in CLASSES:
            continue
        bndbox = element.find('bndbox')
        x1 = int(float(bndbox.find('xmin').text))
        y1 = int(float(bndbox.find('ymin').text))
        x2 = int(float(bndbox.find('xmax').text))
        y2 = int(float(bndbox.find('ymax').text))
        gt.append([org_name, x1, y1, x2, y2])
    return gt


def get_gts(anns_folder):
    gts = {}
    for index, file_name in enumerate(os.listdir(anns_folder)):
        full = os.path.join(anns_folder, file_name)
        gt = convert(full)
        gts[file_name[:file_name.rindex('.')]] = gt
    return gts


def calculate_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def pre_results_pic(results_pic, anns_folder, imgs_folder, out_folder):
    back_data = {}
    for item in results_pic:
        image_name = os.path.basename(item['path'])
        name = image_name[:image_name.rindex('.')]
        if name not in back_data:
            back_data[name] = [item]
        else:
            back_data[name].append(item)
    for xml_name in os.listdir(anns_folder):
        name = xml_name[:xml_name.rindex('.')]
        if name not in back_data:
            back_data[name] = []
    draw_result(imgs_folder, back_data, out_folder)
    return back_data


def draw_result(imgs_folder, results_pic, out_folder):
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    print('draw result to images ...')
    for image_name in os.listdir(imgs_folder):
        name = image_name[:image_name.rindex('.')]
        results = results_pic[name]
        full = os.path.join(imgs_folder, image_name)
        image = cv2.imread(full)
        for box in results:
            cv2.rectangle(image, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])),
                          (0, 0, 255), 3)
            txt = '%s:%s' % (box['type'], box['score'])
            cv2.putText(image, txt, (int(box['xmin']), int(box['ymin'])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(out_folder, image_name), image)
        # print(os.path.join(out_folder, image_name))


def read_pre(file_name, anns_folder, imgs_folder, out_folder):
    with open(file_name) as file:
        lines = file.readlines()
    index = 0
    results = {}
    while index < len(lines):
        line = lines[index]
        if '#' in line:
            org_name = lines[index + 1]
            name = org_name[:org_name.rindex('.')]
            num = int(lines[index + 2])
            results[name] = []
            for i in range(index + 3, index + 3 + num):
                line = lines[i].strip('\r\n').split(' ')
                line = [item for item in line if item != '']
                results[name].append({
                    'xmin': int(float(line[1])),
                    'ymin': int(float(line[2])),
                    'xmax': int(float(line[3])),
                    'ymax': int(float(line[4])),
                    'type': CLASSES[int(line[0])],
                    'score': line[5],
                })
            index += (3 + num)
    for xml_name in os.listdir(anns_folder):
        name = xml_name[:xml_name.rindex('.')]
        if name not in results:
            results[name] = []
    if imgs_folder is not None:
        draw_result(imgs_folder, results, out_folder)
    return results


def acc(anns_folder, imgs_folder, results_file, out_folder, iou_threshold=0.5):
    gts = get_gts(anns_folder, )
    results_pic = read_pre(results_file, anns_folder, imgs_folder, out_folder)
    true_label_num = 0
    gt_num = 0
    gt_pre_class_num = {}
    true_label_pre_class_num = {}
    wrong_pic = 0
    invalid_pic = 0
    miss_pic = 0
    for name in results_pic:
        gts_ = gts[name]
        results = results_pic[name]
        for gt in gts_:
            gt_num += 1
            for result in results:
                iou = calculate_iou([gt[2], gt[1], gt[4], gt[3], ],
                                    [int(result['ymin']), int(result['xmin']), int(result['ymax']),
                                     int(result['xmax'])])
                # 预测正确
                if gt[0] == result['type'] and iou >= iou_threshold:
                    true_label_num += 1
                    if gt[0] not in true_label_pre_class_num:
                        true_label_pre_class_num[gt[0]] = 1
                    else:
                        true_label_pre_class_num[gt[0]] += 1
                    break
            # gt_pre_class数量
            if gt[0] not in gt_pre_class_num:
                gt_pre_class_num[gt[0]] = 1
            else:
                gt_pre_class_num[gt[0]] += 1
        if len(gts_) == 0 and len(results) != 0:
            wrong_pic += 1
        if len(results) > len(gts_) * 100:
            invalid_pic += 1
        if len(gts_) != 0 and len(results) == 0:
            miss_pic += 1

    print('--' * 10)
    for class_name in gt_pre_class_num:
        if class_name not in true_label_pre_class_num:
            true_label_pre_class_num[class_name] = 0
        t = true_label_pre_class_num[class_name] / gt_pre_class_num[class_name]
        print('%s acc: %d/%d = %.3f' % (
            class_name, true_label_pre_class_num[class_name], gt_pre_class_num[class_name], t))
    print('--' * 10)
    print('avg acc : %d/%d = %.3f' % (true_label_num, gt_num, true_label_num / gt_num))
    print('wrong pic : %d/%d = %.3f' % (wrong_pic, len(results_pic), wrong_pic / len(results_pic)))
    print('invalid pic : %d/%d = %.3f' % (invalid_pic, len(results_pic), invalid_pic / len(results_pic)))
    print('miss pic : %d/%d = %.3f' % (miss_pic, len(results_pic), miss_pic / len(results_pic)))


if __name__ == '__main__':
    with open('config.json') as file:
        config = json.load(file)
    CLASSES = config['classes']
    acc(config['anns_folder'], config['imgs_folder'], config['results_file'], config['out_folder'],
        iou_threshold=config['overlap'])
