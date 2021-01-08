import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler

import cv2
import numpy as np
import torch

from planning import *

# import LightDetector
# import pbcvt
# from .CarLightDet import CarLightDet

# DATA_ROOT_DIR = "/home/wanghl/2019-05-09_07-39-38_res.avi"
# DATA_ROOT_DIR = "/home/wanghl/00AD_WEY/2019-11-11_03-25-22_src.avi"
DATA_ROOT_DIR = "/home/wangpeng/2019-11-11_03-25-22_src.avi"


# def update_values(dict_from, dict_to):
#     for key, value in dict_from.items():
#         if isinstance(value, dict):
#             update_values(dict_from[key], dict_to[key])
#         elif value is not None:
#             dict_to[key] = dict_from[key]

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            dict_to[key] = {}
            for key2, value2 in value.items():
                dict_to[key][key2] = value[key2]
            # update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def get_weights(model_path):
    return torch.load(model_path)


def detection_show_result(r_image, obj_dict, color_tl, colors_object,
                          class_names, dis_arr, detect_out):
    num_obj = 0
    for (objectID, bbox_info) in obj_dict.items():
        label = bbox_info["label"]
        # text = "{}".format(label.split(" ")[0])
        text = label
        left, top, right, bottom = bbox_info["bbox"]
        # left = max(left, 0)
        # top = max(top, 0)
        # bottom = min(bottom, r_image.shape[0])
        # right = min(right, r_image.shape[1])
        # print(label)
        if "Trafficlight" in label:
            continue
        elif "FYellow" in label:
            now_draw_color = (0, 0, 0)
            cv2.rectangle(r_image, (left, top), (right, bottom),
                          now_draw_color, 2)
            cv2.putText(
                r_image, "Y_twinkle",
                (bbox_info["centroid"][0] - 10, bbox_info["centroid"][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, now_draw_color, 2)
            cv2.circle(r_image,
                       (bbox_info["centroid"][0], bbox_info["centroid"][1]), 4,
                       now_draw_color, -1)
        else:
            line = ' ' + str(left) + ',' + str(top) + ',' + str(right - left) + ',' + str(bottom - top) + ',' + str(
                label.split(' ')[0])
            detect_out.write(line)
            label = label.strip().split(" ")
            if 'Red' in label[0]:
                now_draw_color = color_tl['Red']
            elif 'Green' in label[0]:
                now_draw_color = color_tl['Green']
            elif 'Yellow' in label[0]:
                now_draw_color = color_tl['Yellow']
            elif 'Off' in label[0]:
                now_draw_color = color_tl['Off']
            else:
                # now_draw_color = colors_object[class_names. (label[0])]
                now_draw_color = colors_object[class_names[label[0]]]
            # txt = '{}{:.1f}'.format(label[0], float(label[1]))
            txt = '{}|{:.1f}|{:.1f}'.format(objectID, dis_arr[num_obj][0], dis_arr[num_obj][1])
            num_obj += 1

            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            cv2.rectangle(r_image, (left, top - cat_size[1] - 2), (left + cat_size[0], top - 2), (0, 255, 0), -1)

            cv2.rectangle(r_image, (int(left), int(top)), (int(right), int(bottom)), now_draw_color, 2)
            cv2.putText(r_image, txt, (left, top - 2), font, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            # cv2.putText(r_image, text,(bbox_info["centroid"][0] - 10, bbox_info["centroid"][1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, now_draw_color, 2)
            # cv2.circle(r_image,
            #            (bbox_info["centroid"][0], bbox_dinfo["centroid"][1]), 4,
            #            now_draw_color, -1)

    # img = np.asarray(r_image)
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Image", r_image)
    # cv2.waitKey(1)


def detection_show(r_image, obj_dict, color_tl, colors_object, class_names, dis_arr):
    num_obj = 0
    for (objectID, bbox_info) in obj_dict.items():
        label = bbox_info["label"]
        # text = "{}".format(label.split(" ")[0])
        text = label
        left, top, right, bottom = bbox_info["bbox"]
        # print(label)
        if "Trafficlight" in label:
            continue
        elif "FYellow" in label:
            now_draw_color = (0, 0, 0)
            cv2.rectangle(r_image, (left, top), (right, bottom),
                          now_draw_color, 2)
            cv2.putText(
                r_image, "Y_twinkle",
                (bbox_info["centroid"][0] - 10, bbox_info["centroid"][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, now_draw_color, 2)
            cv2.circle(r_image,
                       (bbox_info["centroid"][0], bbox_info["centroid"][1]), 4,
                       now_draw_color, -1)
        else:
            label = label.strip().split(" ")
            if 'Red' in label[0]:
                now_draw_color = color_tl['Red']
            elif 'Green' in label[0]:
                now_draw_color = color_tl['Green']
            elif 'Yellow' in label[0]:
                now_draw_color = color_tl['Yellow']
            elif 'Off' in label[0]:
                now_draw_color = color_tl['Off']
            else:
                # now_draw_color = colors_object[class_names.index(label[0])]
                now_draw_color = colors_object[class_names[label[0]]]
            # txt = '{}{:.1f}|{:.1f}|{:.1f}'.format(label[0], float(label[1]), dis_arr[num_obj][0], dis_arr[num_obj][1])
            txt = '{}|{:.1f}|{:.1f}'.format(objectID, dis_arr[num_obj][0], dis_arr[num_obj][1])
            num_obj += 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            # cv2.rectangle(r_image, (left, top - cat_size[1] - 2), (left + cat_size[0], top - 2), (0, 255, 0),-1)

            cv2.rectangle(r_image, (int(left), int(top)), (int(right), int(bottom)), now_draw_color, 2)
            cv2.putText(r_image, txt, (left, top - 2), font, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            # cv2.putText(r_image, text,(bbox_info["centroid"][0] - 10, bbox_info["centroid"][1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, now_draw_color, 2)
            # cv2.circle(r_image,
            #            (bbox_info["centroid"][0], bbox_dinfo["centroid"][1]), 4,
            #            now_draw_color, -1)

    # img = np.asarray(r_image)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", r_image)
    cv2.waitKey(0)


def dec_result_analysis(obj_dict, class_names, age_dict):
    obj_arr_class_all = []
    obj_arr_id_all = []
    pixel_size_arr_all = []
    pixel_central_all = []
    age_count_arr_all = []
    box = []
    for new_tl_keys, items in obj_dict.items():
        left, top, right, bottom = items['bbox']

        temp_label = items['label']
        if "FYellow" in temp_label:
            obj_arr_class_all.append(3)
        else:
            obj_arr_class_all.append(class_names[temp_label.split()[0]])
            # obj_arr_class_all.append(class_names.index(temp_label.split()[0]))
        obj_arr_id_all.append(new_tl_keys)
        pixel_size_arr_all.append([right - left, bottom - top])
        pixel_central_x = left + int((right - left) / 2)
        pixel_central_y = top + int((bottom - top) / 2)
        pixel_central_all.append([pixel_central_x, pixel_central_y])
        age_count_arr_all.append(age_dict[new_tl_keys])
        box.append([left, top, right, bottom])
    return obj_arr_class_all, obj_arr_id_all, age_count_arr_all, pixel_size_arr_all, pixel_central_all, box


def data_lcm_output(dis_arr, size_arr, obj_dict, class_names, age_dict, img):
    class_arr, id_arr, age_count_arr, pixel_size_arr, pixel_central, box = \
        dec_result_analysis(obj_dict, class_names, age_dict)
    # output the traffic light color, relative position to vehicle
    tl_num = 0
    tl_output = CameraMultiObject()

    # left_signal = LeftSignal()
    # left_signal.status = 0

    # left_signal = CarLightDet()

    for ind, dis in enumerate(dis_arr):
        dx, dy = dis[0], dis[1]
        tl_num += 1
        pos = Point()
        size_r = Point()
        size_p = Point()
        central_p = Point()
        pos.x, pos.y = int(dx), int(dy)
        size_r.y, size_r.x = int(size_arr[ind][0]), int(size_arr[ind][1])
        size_p.x, size_p.y = int(pixel_size_arr[ind][0]), int(
            pixel_size_arr[ind][1])
        central_p.x, central_p.y = int(pixel_central[ind][0]), int(
            pixel_central[ind][1])
        left, top, right, bottom = int(box[ind][0]), int(box[ind][1]), int(
            box[ind][2]), int(box[ind][3])

        s_object = CameraObject()
        s_object.relPosition = pos

        s_object.object = class_arr[ind]  # number
        s_object.object_id = id_arr[ind]
        s_object.ageCount = age_count_arr[ind]
        s_object.size = size_r
        s_object.PixelSize = size_p
        s_object.PixelCentral = central_p
        # tl_output.Objects.append(s_object)
        if (s_object.object < 17 and s_object.ageCount < 3):
            tl_num = tl_num - 1
        elif s_object.object == 27 or s_object.object == 28:
            tl_num = tl_num - 1
        else:
            tl_output.Objects.append(s_object)
        tl_output.num = tl_num

    return tl_output


def data_lcm_output1(dis_arr, size_arr, obj_dict, class_names, age_dict, img):
    class_arr, id_arr, age_count_arr, pixel_size_arr, pixel_central, box = \
        dec_result_analysis(obj_dict, class_names, age_dict)
    # output the traffic light color, relative position to vehicle
    tl_num = 0
    tl_output = CameraMultiObject()

    left_signal = CarLightDet()
    left_signal.status = 0
    for ind, dis in enumerate(dis_arr):
        dx, dy = dis[0], dis[1]
        tl_num += 1
        pos = Point()
        size_r = Point()
        size_p = Point()
        central_p = Point()
        pos.x, pos.y = int(dx), int(dy)
        # print("+++++++++", pos.x, "+++++++++", pos.y, "+++++++++", dx, "+++++++++", dy)
        size_r.y, size_r.x = int(size_arr[ind][0]), int(size_arr[ind][1])
        size_p.x, size_p.y = int(pixel_size_arr[ind][0]), int(
            pixel_size_arr[ind][1])
        central_p.x, central_p.y = int(pixel_central[ind][0]), int(
            pixel_central[ind][1])
        left, top, right, bottom = int(box[ind][0]), int(box[ind][1]), int(
            box[ind][2]), int(box[ind][3])

        s_object = CameraObject()
        s_object.relPosition = pos
        # if class_arr[ind] in [5, 6, 7]:
        #     class_arr[ind] = class_arr[ind] - 5
        #     if class_arr[ind] == 0:
        #         class_arr[ind] = 1
        #     elif class_arr[ind] == 1:
        #         class_arr[ind] = 0

        if (class_arr[ind] == 19):
            # print("jianche:", left, top, right, bottom)
            # print("jianche:", left, top, right, bottom)
            # print(central_p.x, central_p.y)
            # print(img.shape[1] * 0.55, img.shape[1] * 0.9, img.shape[0] * 0.4,
            #       img.shape[0] * 0.95)
            weight = right - left

            # if ((central_p.x > img.shape[1] * 0.55
            #      and central_p.x < img.shape[1] * 0.95)
            #         and (central_p.y > img.shape[0] * 0.45
            #              and central_p.y < img.shape[0] * 0.95)):
            boundingBox_weight = right + 0.2 * weight
            if boundingBox_weight >= img.shape[1] - 1:
                boundingBox_weight = img.shape[1] - 1
            car = img[top:bottom, left:int(boundingBox_weight)]

            # flag_left = isLeftLight_fc(car)
            # flag_left = isLeft(car)
            # ld = LightDetector.LightDetector()
            flag_left = left_signal.detectCarLight(car)
            # flag_left = ld.detect_py(car)

            print("*" * 60, flag_left)
            if (flag_left):
                left_signal.status = 1
            # if((central_p.x > img.shape[1]*0.55 and central_p.x  < img.shape[1]*0.95)
            #    and (central_p.y > img.shape[0]*0.45 and central_p.y < img.shape[0]*0.95)):
            #     if((int(top+0.1*height) < int(bottom-height*0.2)) and (int(left+weight*0.15) < int(right -weight*0.35))):
            #         car = img[int((top+0.1*height)):int((bottom-height*0.2)),int(left+weight*0.15):int((right-weight*0.35))]
            #     else:
            #         print("-----------------------------------------------------")
            #         car = img[top:bottom,left:right]
            #     flag_left = isLeftLight_fc(car)
            #     #flag_left = isLeft(car)
            #     print("*"*60, flag_left)
            #     if (flag_left):
            #         left_signal.status = 1

        s_object.object = class_arr[ind]
        s_object.object_id = id_arr[ind]
        s_object.ageCount = age_count_arr[ind]
        s_object.size = size_r
        s_object.PixelSize = size_p
        s_object.PixelCentral = central_p

        # tl_output.Objects.append(s_object) if BDD detect traffic lights then delete from lcm
        if s_object.object_id != 23:
            tl_output.Objects.append(s_object)
        else:
            tl_num = tl_num - 1
    tl_output.num = tl_num
    return tl_output  # , left_signal


def isLeft(image):
    color = [([50, 120, 120], [120, 170, 170])]
    print(image.shape[0], image.shape[1], image.shape[2])

    for (lower, upper) in color:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        index = np.where(mask == 255)
        if (len(index[0]) > 0):
            pos_x = int(np.sum(index[0]) / len(index[0]))
            pos_y = int(np.sum(index[1]) / len(index[1]))
            print("left_pos:", pos_x, pos_y)
            print(image.shape[1] * 0.1, image.shape[1] * 0.7,
                  image.shape[0] * 0.1, image.shape[0] * 0.9)
            if ((pos_x > image.shape[0] * 0.1 and pos_x < image.shape[0] * 0.7)
                    and (pos_y > image.shape[1] * 0.1
                         and pos_y < image.shape[1] * 0.7)):
                return True
        else:
            return False
        # 增加判断逻辑


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_dir = os.path.join(os.getcwd(), "log")
    if not os.path.exists(log_dir):
        print("Save log to:", log_dir)
        os.mkdir(log_dir)
    # 创建一个handler，用于写入日志文件
    log_name = "{}_log.txt".format(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())))
    # fh = logging.FileHandler(os.path.join(log_dir, log_name), mode="w")
    # fh.setLevel(logging.DEBUG)
    fh = TimedRotatingFileHandler(filename=os.path.join(log_dir, log_name), when="H", backupCount=24, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter(
        '%(asctime)s - %(module)s.%(funcName)s.%(lineno)d - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    # 记录一条日志
    logger.info('................Object Detection Start!')
    return logger


detection_logger = init_logger()


class LoadImg(object):
    def __init__(self):
        self.video_path = DATA_ROOT_DIR
        self.img = self.video2img()

    def video2img(self):
        video_capture = cv2.VideoCapture(self.video_path)
        if video_capture.isOpened():
            _, frame = video_capture.read()
            while True:
                _, frame = video_capture.read()
                yield frame

    def load_img(self):
        return next(self.img)
