from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import torch
import random
import pickle
import cv2 as cv

class DeepFashionDataset(Dataset):
    def __init__(self, path, stage1_dir, phase, size, all_view=False, vol_feat_res=128):
        # path = /home/nas1_temp/dataset/Thuman
        self.path = path
        self.stage1_dir = stage1_dir
        self.phase = phase  # train or test
        self.size = size    # 256 or 512  FOR  174x256 or 348x512
        self.vol_feat_res = vol_feat_res
        if all_view:
            self.target_view_diff_list = [90, 180, 270]
        else:
            self.target_view_diff_list = [180]

        # initialize the pairs of data
        self.init_pairs()
        self.data_size = len(self.pairs)
        print('%s data pairs (#=%d)...'%(phase, self.data_size))

        # if phase == 'train':
        #     # get dictionary of image name and transfrom to detect and align the face
        #     with open(os.path.join(path, 'resources', 'train_face_T.pickle'), 'rb') as handle:
        #         self.faceTransform = pickle.load(handle)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.totensor = transforms.ToTensor()

    def init_pairs(self) :#, pairs_csv_path):
        # pairs_file = pd.read_csv(pairs_csv_path)
        self.pairs = []
        self.sources = {}
        print('Loading data pairs ...')
        # for i in range(len(pairs_file)):
        #     pair = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
        #     self.pairs.append(pair)


        if self.phase == 'train':
            model_id_list = []
            with open(os.path.join(self.path,'data_list_train.txt')) as f:
                id = f.readlines()
            model_id_list = list(map(lambda x: x[:-1] if x.endswith('\n') else x, id))

            for model_id in model_id_list:
                for source_view_id in list(range(0, 360, 18)):
                    for target_veiw_diff in [180]:
                        target_view_id = target_veiw_diff + source_view_id
                        if target_view_id >= 360:
                            target_view_id = target_view_id - 360
                        pair = [model_id, source_view_id, target_view_id]
                        self.pairs.append(pair)

        elif self.phase == 'test':
            # for i in range(501, 526):
            #     model_id = str(i).zfill(4)
            #     for source_view_id in [0, 180]:
            #         for target_view_diff in [180]:
            #             target_view_id = target_view_diff + source_view_id
            #             if target_view_id >= 360:
            #                 target_view_id = target_view_id - 360
            #             pair = [model_id, source_view_id, target_view_id]
            #             self.pairs.append(pair)

            # source_view_list = [138, 155, 195, 73, 303, 225, 240, 333, 136, 197, 222, 272, 291, 298, 147, 38, 194,
            #                     275, 348, 40, 1, 13, 325, 273, 186]
            # for num, i in enumerate(range(501,526)):
            #     model_id = str(i).zfill(4)
            #     source_view_id = source_view_list[num]
            #     target_view_id = source_view_id + 180
            #     if target_view_id >= 360:
            #         target_view_id = target_view_id - 360
            #     pair = [model_id, source_view_id, target_view_id]
            #     self.pairs.append(pair)

            # source_view_list = [0, 90, 180, 270]
            # for num, i in enumerate(range(501, 526)):
            #     model_id = str(i).zfill(4)
            #     for source_view_id in source_view_list:
            #         target_view_id = source_view_id + 180
            #         if target_view_id >= 360:
            #             target_view_id = target_view_id - 360
            #         pair = [model_id, source_view_id, target_view_id]
            #         self.pairs.append(pair)

            if 'thuman' in self.stage1_dir:
                data_name = 'thuman'
            elif 'twindom' in self.stage1_dir:
                data_name = 'twindom'
            else:
                raise NotImplementedError()

            with open(os.path.join(self.path,f'data_list_test_{data_name}.txt')) as f:
                id = f.readlines()
            model_id_list = list(map(lambda x: x[:-1] if x.endswith('\n') else x, id))

            for model_id in model_id_list:
                for source_view_id in [0, 90, 180, 270]:
                    for target_veiw_diff in [180]:
                        target_view_id = target_veiw_diff + source_view_id
                        if target_view_id >= 360:
                            target_view_id = target_view_id - 360
                        pair = [model_id, source_view_id, target_view_id]
                        self.pairs.append(pair)


        elif self.phase == 'val':
            with open(os.path.join(self.path,'data_list_test.txt')) as f:
                id = f.readlines()
            model_id_list = list(map(lambda x: x[:-1] if x.endswith('\n') else x, id))
            self.pairs.append([model_id_list[0], 0, 180])
            self.pairs.append([model_id_list[1], 0, 180])
            self.pairs.append([model_id_list[2], 0, 180])
            self.pairs.append([model_id_list[3], 180, 0])
            # self.pairs.append([model_id_list[4], 180, 0])
            # self.pairs.append([model_id_list[5], 180, 0])
            self.pairs.append([model_id_list[-1], 0, 180])
            self.pairs.append([model_id_list[-2], 0, 180])
            self.pairs.append([model_id_list[-3], 0, 180])
            self.pairs.append([model_id_list[-4], 180, 0])
            # self.pairs.append([model_id_list[-5], 180, 0])
            # self.pairs.append([model_id_list[-6], 180, 0])


        print('Loading data pairs finished ...')


    def __len__(self):
        return self.data_size


    def resize_height_PIL(self, x, height=512):
        w, h = x.size
        width  = int(height * w / h)
        return x.resize((width, height), Image.NEAREST) #Image.ANTIALIAS


    def resize_PIL(self, x, height=512, width=348, type=Image.NEAREST):
        return x.resize((width, height), type)


    def tensors2square(self, im, pose, sil):
        width = im.shape[2]
        diff = self.size - width
        if self.phase == 'train':
            left = random.randint(0, diff)
            right = diff - left
        else: # when testing put in the center
            left = int((self.size-width)/2)
            right = diff - left
        im = torch.nn.functional.pad(input=im, pad=(right, left, 0, 0), mode='constant', value=0)
        pose = torch.nn.functional.pad(input=pose, pad=(right, left, 0, 0), mode='constant', value=0)
        sil = torch.nn.functional.pad(input=sil, pad=(right, left, 0, 0), mode='constant', value=0)
        return im, pose, sil, left, right

    def load_image(self, data_item, view_id):
        # img_fpath = os.path.join(
        #     self.path, 'image_data_nolight2', str(data_item).zfill(4), 'color_re/%04d.jpg' % view_id)
        img_fpath = os.path.join(
            self.path, 'image_data', str(data_item).zfill(4), 'color/%04d.jpg' % view_id)
        msk_fpath = os.path.join(
            self.path, 'image_data', str(data_item).zfill(4), 'mask/%04d.png' % view_id)
        try:
            img = Image.open(img_fpath).convert('RGB')
            msk = Image.open(msk_fpath)
        except:
            raise RuntimeError('Failed to load iamge: ' + img_fpath)

        #todo msk 벨류 확인
        return img, msk

    def load_stage1_output(self, data_item, source_view_id, target_view_id, vol_feat_res):
        # if self.phase == 'train' or self.phase == 'val':
        #     stage1_dir = 'pamir_nerf_0222_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie'
        #
        # elif self.phase == 'test':
        #     stage1_dir = 'pamir_nerf_0222_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie_val_4view_0304'

        flow_fpath = os.path.join(
            self.stage1_dir, str(data_item).zfill(4),
            'flow/%04d_%04d.png' % (source_view_id, target_view_id))
        pred_image_fpath = os.path.join(
            self.stage1_dir, str(data_item).zfill(4),
            'pred_image/%04d_%04d.png' % (source_view_id, target_view_id))
        attention_fpath = os.path.join(
            self.stage1_dir, str(data_item).zfill(4),
            'attention/%04d_%04d.png' % (source_view_id, target_view_id))
        target_msk_fpath = os.path.join(
            self.stage1_dir, str(data_item).zfill(4),
            'weight_sum/%04d_%04d.png' % (source_view_id, target_view_id))


        # try:
        flow = Image.open(flow_fpath)
        pred_image = Image.open(pred_image_fpath)
        # attention = Image.open(attention_fpath)
        target_msk = Image.open(target_msk_fpath)
        # except:
        #     raise RuntimeError('Failed to load stage1 output: ' + flow_fpath)
        return flow, pred_image, None, target_msk


    def __getitem__(self, index):
        # get current pair
        # im1_name, im2_name = self.pairs[index]
        model_id, source_view_id, target_view_id = self.pairs[index]

        if self.phase == 'test':
            # set target height and target width
            input_image, silhouette1 = self.load_image(model_id, source_view_id)
            target_image, silhouette2 = self.load_image(model_id, target_view_id)
            flow, pred_image, attention, target_msk = self.load_stage1_output(model_id, source_view_id, target_view_id, self.vol_feat_res)
            silhouette2 = target_msk
        else:
            input_image, silhouette1 = self.load_image(model_id, source_view_id)
            target_image, silhouette2 = self.load_image(model_id, target_view_id)
            flow, pred_image, attention, target_msk = self.load_stage1_output(model_id, source_view_id, target_view_id, self.vol_feat_res)

        # read uv-space data
        # complete_coor = np.load(complete_coor_path)

        # Transform
        input_image = self.transform(input_image)
        target_image = self.transform(target_image)
        flow = self.transform(flow)[:2]
        pred_image = self.transform(pred_image)
        silhouette1 = self.totensor(silhouette1)
        silhouette2 = self.totensor(silhouette2)
        # attention = self.totensor(attention)[:1]

        # put into a square
        input_image, _, silhouette1, Sleft, Sright = self.tensors2square(input_image, flow, silhouette1)
        target_image, flow, silhouette2, Tleft, Tright = self.tensors2square(target_image, flow, silhouette2)
        if Sleft != 0 and Tleft != 0:
            raise NotImplementedError()


        target_im = target_image
        target_sil = silhouette2
        target_image_name = target_view_id
        target_left_pad = Tleft
        target_right_pad = Tright

        FT = torch.zeros((3,3))

        # return data
        if self.phase == 'train':
            save_name = str(model_id).zfill(4) + '_' + str(source_view_id).zfill(4) + '_2_' + str(source_view_id).zfill(
                4) + '_vis.png'
            return {'input_image':input_image, 'target_image':target_im, 'pred_image': pred_image, # 'attention': attention,
                    'target_sil': target_sil,
                    'flow': flow,
                    'TargetFaceTransform': FT,
                    'target_left_pad':torch.tensor(target_left_pad),
                    'target_right_pad':torch.tensor(target_right_pad),
                    'input_sil': silhouette1,
                    'save_name': save_name,
                    'model_id': model_id
                    }

        else:
            save_name = str(model_id).zfill(4) + '_' + str(source_view_id).zfill(4) + '_2_' + str(source_view_id).zfill(4) + '_vis.png'
            return {'input_image':input_image, 'target_image':target_im, 'pred_image': pred_image, # 'attention': attention,
                    'target_sil': target_sil,
                    'target_left_pad':torch.tensor(target_left_pad),
                    'target_right_pad':torch.tensor(target_right_pad),
                    'flow': flow,
                    'input_sil': silhouette1,
                    'save_name':save_name,
                    'model_id': model_id,
                    'source_view_id': str(source_view_id).zfill(4),
                    'target_view_id': str(target_view_id).zfill(4)
                    }



class InferenceDataset(Dataset):
    def __init__(self, path, size, vol_feat_res=128):
        # path = /home/nas1_temp/dataset/Thuman
        self.path = path
        self.size = size    # 256 or 512  FOR  174x256 or 348x512
        self.vol_feat_res = vol_feat_res

        self.target_view_diff_list = [180]

        # initialize the pairs of data
        self.init_pairs()
        self.data_size = len(self.pairs)
        print('data pairs (#=%d)...'%(self.data_size))

        # if phase == 'train':
        #     # get dictionary of image name and transfrom to detect and align the face
        #     with open(os.path.join(path, 'resources', 'train_face_T.pickle'), 'rb') as handle:
        #         self.faceTransform = pickle.load(handle)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.totensor = transforms.ToTensor()

    def init_pairs(self) :#, pairs_csv_path):
        # pairs_file = pd.read_csv(pairs_csv_path)
        self.pairs = []
        self.sources = {}
        print('Loading data pairs ...')
        # for i in range(len(pairs_file)):
        #     pair = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
        #     self.pairs.append(pair)
        model_id_list = os.listdir(os.path.join(self.path, 'output_stage1'))
        for model_id in model_id_list:
            source_view_id = 0
            target_view_id = source_view_id + 180
            if target_view_id >= 360:
                target_view_id = target_view_id - 360
            pair = [model_id, source_view_id, target_view_id]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')


    def __len__(self):
        return self.data_size


    def resize_height_PIL(self, x, height=512):
        w, h = x.size
        width  = int(height * w / h)
        return x.resize((width, height), Image.NEAREST) #Image.ANTIALIAS


    def resize_PIL(self, x, height=512, width=348, type=Image.NEAREST):
        return x.resize((width, height), type)


    def tensors2square(self, im, pose, sil):
        width = im.shape[2]
        diff = self.size - width

        left = int((self.size-width)/2)
        right = diff - left
        im = torch.nn.functional.pad(input=im, pad=(right, left, 0, 0), mode='constant', value=0)
        pose = torch.nn.functional.pad(input=pose, pad=(right, left, 0, 0), mode='constant', value=0)
        sil = torch.nn.functional.pad(input=sil, pad=(right, left, 0, 0), mode='constant', value=0)
        return im, pose, sil, left, right

    def load_image(self, data_item):
        img_fpath = os.path.join(
            self.path, '../', 'image', data_item + '.jpg')
        msk_fpath = os.path.join(
            self.path, '../', 'image', data_item + '_mask.png')

        try:
            img = Image.open(img_fpath).convert('RGB')
            msk = Image.open(msk_fpath)
        except:
            raise RuntimeError('Failed to load iamge: ' + img_fpath)

        #todo msk 벨류 확인
        return img, msk

    def load_stage1_output(self, data_item, source_view_id, target_view_id):
        flow_fpath = os.path.join(
            self.path, 'output_stage1', data_item,
            'flow/%04d_%04d.png' % (source_view_id, target_view_id))
        pred_image_fpath = os.path.join(
            self.path, 'output_stage1', data_item,
            'pred_image/%04d_%04d.png' % (source_view_id, target_view_id))
        attention_fpath = os.path.join(
            self.path, 'output_stage1', data_item,
            'attention/%04d_%04d.png' % (source_view_id, target_view_id))
        target_msk_fpath = os.path.join(
            self.path, 'output_stage1', data_item,
            'weight_sum/%04d_%04d.png' % (source_view_id, target_view_id))

        try:
            flow = Image.open(flow_fpath)
            pred_image = Image.open(pred_image_fpath)
            # attention = Image.open(attention_fpath)
            target_msk = Image.open(target_msk_fpath)
        except:
            raise RuntimeError('Failed to load stage1 output: ' + flow_fpath)
        return flow, pred_image, None, target_msk


    def __getitem__(self, index):
        # get current pair
        # im1_name, im2_name = self.pairs[index]
        model_id, source_view_id, target_view_id = self.pairs[index]


        input_image, silhouette1 = self.load_image(model_id)
        flow, pred_image, attention, silhouette2 = self.load_stage1_output(model_id, source_view_id, target_view_id)

        # Transform
        input_image = self.transform(input_image)
        flow = self.transform(flow)[:2]
        pred_image = self.transform(pred_image)
        silhouette1 = self.totensor(silhouette1)
        silhouette2 = self.totensor(silhouette2)
        # attention = self.totensor(attention)[:1]

        # put into a square
        input_image, _, silhouette1, Sleft, Sright = self.tensors2square(input_image, flow, silhouette1)

        target_sil = silhouette2
        target_image_name = target_view_id

        FT = torch.zeros((3,3))

        # return data



        save_name = str(model_id).zfill(4) + '_' + str(source_view_id).zfill(4) + '_2_' + str(source_view_id).zfill(4) + '_vis.png'
        return {'input_image':input_image, 'pred_image': pred_image, # 'attention': attention,
                'target_sil': target_sil,
                'flow': flow,
                'input_sil': silhouette1,
                'save_name':save_name,
                'model_id': model_id,
                'source_view_id': str(source_view_id).zfill(4),
                'target_view_id': str(target_view_id).zfill(4)
                }