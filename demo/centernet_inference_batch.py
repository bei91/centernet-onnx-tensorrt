# -*-coding:utf-8-*-
import time

import cv2
import numpy as np
import tensorrt as trt
import torch

import yaml

from network.decode import ctdet_decode, ctdet_post_process
from utils.image import get_affine_transform
from utils.utils import update_values
from .detection import DetectionObj
from .trt_function import do_inference, allocate_buffers
from .trt_inference import TrtInference, Singleton
from scipy.spatial import distance as dist



@Singleton
class CenterNetTrtInferenceBatch(TrtInference):

    def __init__(self, opts):
        super().__init__()
        with open("./" + opts.cfg_path_tl, 'r') as handle:
            self.options_yaml_tl = yaml.load(handle)
        update_values(self.options_yaml_tl, vars(opts))
        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100

        self.opt = opts
        self.net_inp_height, self.net_inp_width = self.opt.height, self.opt.width
        self.net_out_height, self.net_out_width = self.net_inp_height // 4, self.net_inp_width // 4
        self.num_classes = len(self.opt.classes_names)

        self.m_obj_engine = self.load_engine()
        self.detection_inputs, self.detection_outputs, self.detection_bindings, self.detection_stream = allocate_buffers(
            self.m_obj_engine)
        self.detection_context = self.m_obj_engine.create_execution_context()

    @property
    def __meta_c(self):
        return np.array([self.image_width / 2., self.image_height / 2.], dtype=np.float32)

    @property
    def __meta_s(self):
        return max(self.image_height, self.image_width) * 1.0

    def load_engine(self):
        engine_path = self.opt.day_engine_path if self.opt.day_or_night else self.opt.night_engine_path
        with open(engine_path, "rb") as obj_f, trt.Runtime(TrtInference.TRT_LOGGER) as runtime:
            obj_engine = runtime.deserialize_cuda_engine(obj_f.read())
        return obj_engine

    def pre_process(self, img_batch):
        imgs_infer = []
        for image_raw in img_batch:
            image = cv2.imread(image_raw)
            self.image_height, self.image_width = image.shape[0:2]
            trans_input = get_affine_transform(self.__meta_c, self.__meta_s, 0, [self.net_inp_width, self.net_inp_height])
            inp_image = cv2.warpAffine(image, trans_input, (self.net_inp_width, self.net_inp_height),
                                       flags=cv2.INTER_LINEAR)

            inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
            images = inp_image.transpose(2, 0, 1).reshape(1, 3, self.net_inp_height, self.net_inp_width)
            imgs_infer.append(images)
        if len(img_batch) == 1:
            images_batch = images
        else:
            images_batch = imgs_infer[0]
            for i in range(len(imgs_infer)-1):
                images_batch = np.concatenate((images_batch, imgs_infer[i+1]))
        img_numpy = np.ascontiguousarray(images_batch, dtype=np.float32)
        return img_numpy

    def infer(self, img_batch, pre_result=None):
        infer_time = time.time()
        image = self.pre_process(img_batch)
        output = self.__trt_infer(image)
        results_batch = self.__bbox_decode(output)
        objlist_batch = self.__convert(results_batch, infer_time)
        reslist_batch = self.__filter_same_bbox(objlist_batch)
        return reslist_batch

    def __convert(self, results_batch, infer_time):
        objs_batch = []
        for i in range(len(results_batch)):
            results = results_batch[i]
            objlist = []
            for j in range(1, self.num_classes + 1):
                for bbox in results[j]:
                    bbox_dict = dict()
                    score = bbox[4]
                    if score > self.opt.tl_threshold_low:
                        categrey_id = j - 1
                        bbox_dict['timestamp'] = infer_time
                        bbox_dict["class_name"] = self.opt.classes_names[categrey_id]
                        bbox_dict["class_id"] = categrey_id
                        bbox_dict["topleft_x"] = max(0, bbox[0])
                        bbox_dict["topleft_y"] = max(0, bbox[1])
                        bbox_dict["bottomright_x"] = min(self.image_width, bbox[2])
                        bbox_dict["bottomright_y"] = min(self.image_height, bbox[3])
                        bbox_dict["score"] = score
                        obj = DetectionObj(**bbox_dict)
                        objlist.append(obj)
            objs_batch.append(objlist)
        return objs_batch

    def __filter_same_bbox(self, obj_list_batch):
        res_list_batch = []
        for i in range(len(obj_list_batch)):
            obj_list = obj_list_batch[i]
            res_list = []
            if len(obj_list) == 0:
                return res_list

            match_dict = {}
            obj_list_centroids = [
                (obj.topleft_x + (obj.bottomright_x - obj.topleft_x) / 2,
                 obj.topleft_y + (obj.bottomright_y - obj.topleft_y) / 2) for obj in obj_list
            ]
            obj_distance = dist.cdist(np.array(obj_list_centroids), np.array(obj_list_centroids))
            min_dis_index = np.argsort(obj_distance)
            for i in range(len(obj_list)):
                if min_dis_index[i][0] in match_dict:
                    match_dict[min_dis_index[i][0]].append(i)
                else:
                    match_dict[min_dis_index[i][0]] = [i]
            for k, v in match_dict.items():
                if len(v) == 1 and k == v[0]:
                    res_list.append(obj_list[k])
                else:
                    score_list = []
                    for i in v:
                        score_list.append(obj_list[i].score)
                    max_index = score_list.index(max(score_list))
                    last_obj_index = v[max_index]
                    res_list.append(obj_list[last_obj_index])
            res_list_batch.append(res_list)
        return res_list_batch

    def __trt_infer(self, image):
        print(self.detection_inputs[0].host.shape, image.shape)  #(12582912,) (4, 3, 512, 512)
        self.detection_inputs[0].host = image
        # np.copyto(self.detection_inputs[0].host, image.ravel()) #ValueError: could not broadcast input array from shape (3145728) into shape (12582912)
        output = do_inference(
            context=self.detection_context,
            bindings=self.detection_bindings,
            inputs=self.detection_inputs,
            outputs=self.detection_outputs,
            stream=self.detection_stream,
            batch_size=4)

        return output

    def __bbox_decode(self, infer_out):
        hm_batch = []
        wh_batch = []
        reg_batch = []
        results_batch = []
        patch_ratio = 4   # when tensorrt7 is 4, other version could be 1
        batch_size = 4
        for i in range(0, len(infer_out[0]), int(len(infer_out[0])/(batch_size * batch_size))):
            if i * patch_ratio < int(len(infer_out[0])):
                hm = torch.from_numpy(infer_out[0][i:int(len(infer_out[0]) / (4 * patch_ratio))+i]).sigmoid_()
                hm = hm.reshape(1, self.num_classes, self.net_out_height, self.net_out_width)
                hm_batch.append(hm)
        for i in range(0, len(infer_out[1]), int(len(infer_out[1]) / (batch_size * batch_size))):
            if i * patch_ratio < int(len(infer_out[1])):
                wh = torch.from_numpy(infer_out[1][i:int(len(infer_out[1]) / (batch_size * patch_ratio))+i])
                wh = wh.reshape(1, 2, self.net_out_height,   self.net_out_width)
                wh_batch.append(wh)
        for i in range(0, len(infer_out[2]), int(len(infer_out[2]) / (batch_size * batch_size))):
            if i * patch_ratio < int(len(infer_out[2])):
                reg = torch.from_numpy(infer_out[2][i:int(len(infer_out[2]) / (batch_size * patch_ratio))+i])
                reg = reg.reshape(1, 2, self.net_out_height, self.net_out_width)
                reg_batch.append(reg)
        for i in range(len(reg_batch)):
            dets = ctdet_decode(hm_batch[i], wh_batch[i], reg=reg_batch[i], K=100)
            dets = self.__box_post_process(dets)
            results = self.__merge_outputs([dets])
            results_batch.append(results)
        return results_batch

    def __box_post_process(self, dets):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [self.__meta_c], [self.__meta_s], self.net_out_height,
                                  self.net_out_width,
                                  self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= 1
        return dets[0]

    def __merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)
            # soft_nms(results[j], Nt=0.5, method=2)

        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])

        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results
