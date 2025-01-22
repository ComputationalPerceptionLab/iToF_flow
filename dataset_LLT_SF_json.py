import os
import cv2
import ast
import torch
import numpy as np
import random
import imageio
from torch.utils.data import DataLoader, Dataset
import json

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ToFDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 480
        self.w = 640
        self.data_root = '/xxxx/ToF_AFLLT_SF_json/'
        print('Loading data from {}'.format(self.data_root))
        self.image_root = os.path.join(self.data_root, 'sequence')
        self.DATA_PATH = ''
        train_fn = os.path.join(self.data_root, 'train_motion.txt')
        test_fn = os.path.join(self.data_root, 'val_motion.txt')
        plot_fn = os.path.join(self.data_root, 'val_motion.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        with open(plot_fn, 'r') as f:
            self.plotlist = f.read().splitlines()
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        elif self.dataset_name == 'plot':
            self.meta_data = self.plotlist
        else:
            self.meta_data = self.trainlist[cnt:]
           
    def crop(self,  img_00, img_11, img_22, img_33, img_40, img_40_gt, img_41_gt, img_42_gt, img_43_gt, img_00_LLT, img_01_LLT, img_02_LLT, img_03_LLT, freq1, freq2, x,y, h, w):

        img_00 = img_00[x:x+h, y:y+w]
        img_11 = img_11[x:x+h, y:y+w]
        img_22 = img_22[x:x+h, y:y+w]
        img_33 = img_33[x:x+h, y:y+w]
        
        img_40 = img_40[x:x+h, y:y+w]

        img_40_gt = img_40_gt[x:x+h, y:y+w]
        img_41_gt = img_41_gt[x:x+h, y:y+w]
        img_42_gt = img_42_gt[x:x+h, y:y+w]
        img_43_gt = img_43_gt[x:x+h, y:y+w]

        img_00_LLT = img_00_LLT[x:x+h, y:y+w]
        img_01_LLT = img_01_LLT[x:x+h, y:y+w]
        img_02_LLT = img_02_LLT[x:x+h, y:y+w]
        img_03_LLT = img_03_LLT[x:x+h, y:y+w]

        freq1 = freq1[x:x+h, y:y+w]
        freq2 = freq2[x:x+h, y:y+w]

        return img_00, img_11, img_22, img_33, img_40, img_40_gt, img_41_gt, img_42_gt, img_43_gt, img_00_LLT, img_01_LLT, img_02_LLT, img_03_LLT, freq1, freq2

    def getimg(self, index):
        json_path = os.path.join(self.image_root, self.meta_data[index])
        with open(json_path, 'r') as f:

            content = f.read()
            data = json.loads(content)
            img_00_path_array = data.get('img_00_path_array')
            img_11_path_array = data.get('img_11_path_array')
            img_22_path_array = data.get('img_22_path_array')
            img_33_path_array = data.get('img_33_path_array')

            img_40_path_array = data.get('img_40_path_array')

            img_40_gt_path_array = data.get('img_40_gt_path_array')
            img_41_gt_path_array = data.get('img_41_gt_path_array')
            img_42_gt_path_array = data.get('img_42_gt_path_array')
            img_43_gt_path_array = data.get('img_43_gt_path_array')

            img_00_LLT_path_array = data.get('img_00_LLT_path_array')
            img_01_LLT_path_array = data.get('img_01_LLT_path_array')
            img_02_LLT_path_array = data.get('img_02_LLT_path_array')
            img_03_LLT_path_array = data.get('img_03_LLT_path_array')

            frequency1 = data.get('freq')
            frequency2 = data.get('freq')

        

        res_tensor = []
        for i in range(len(img_00_path_array)):
            img_00_path = img_00_path_array[i]
            img_11_path = img_11_path_array[i]
            img_22_path = img_22_path_array[i]
            img_33_path = img_33_path_array[i]

            img_40_path = img_40_path_array[i]

            img_40_gt_path = img_40_gt_path_array[i]
            img_41_gt_path = img_41_gt_path_array[i]
            img_42_gt_path = img_42_gt_path_array[i]
            img_43_gt_path = img_43_gt_path_array[i]

            img_00_LLT_path = img_00_LLT_path_array[i]
            img_01_LLT_path = img_01_LLT_path_array[i]
            img_02_LLT_path = img_02_LLT_path_array[i]
            img_03_LLT_path = img_03_LLT_path_array[i]


            img_00 = imageio.imread(os.path.join(self.DATA_PATH, img_00_path), format='HDR-FI')[:, :, 0]
            img_11 = imageio.imread(os.path.join(self.DATA_PATH, img_11_path), format='HDR-FI')[:, :, 0]
            img_22 = imageio.imread(os.path.join(self.DATA_PATH, img_22_path), format='HDR-FI')[:, :, 0]
            img_33 = imageio.imread(os.path.join(self.DATA_PATH, img_33_path), format='HDR-FI')[:, :, 0]
            
            img_40 = imageio.imread(os.path.join(self.DATA_PATH, img_40_path), format='HDR-FI')[:, :, 0]

            img_40_gt = imageio.imread(os.path.join(self.DATA_PATH, img_40_gt_path), format='HDR-FI')[:, :, 0]
            img_41_gt = imageio.imread(os.path.join(self.DATA_PATH, img_41_gt_path), format='HDR-FI')[:, :, 0]
            img_42_gt = imageio.imread(os.path.join(self.DATA_PATH, img_42_gt_path), format='HDR-FI')[:, :, 0]
            img_43_gt = imageio.imread(os.path.join(self.DATA_PATH, img_43_gt_path), format='HDR-FI')[:, :, 0]

            img_00_LLT = imageio.imread(os.path.join(self.DATA_PATH, img_00_LLT_path), format='HDR-FI')[:, :, 0]
            img_01_LLT = imageio.imread(os.path.join(self.DATA_PATH, img_01_LLT_path), format='HDR-FI')[:, :, 0]
            img_02_LLT = imageio.imread(os.path.join(self.DATA_PATH, img_02_LLT_path), format='HDR-FI')[:, :, 0]
            img_03_LLT = imageio.imread(os.path.join(self.DATA_PATH, img_03_LLT_path), format='HDR-FI')[:, :, 0]



            self.h, self.w = img_00.shape
            if self.h < 600:
                max_value = 1024
            else:
                max_value = 4000

            img_00[img_00 > max_value] = max_value
            img_11[img_11 > max_value] = max_value
            img_22[img_22 > max_value] = max_value
            img_33[img_33 > max_value] = max_value

            img_40[img_40 > max_value] = max_value

            img_40_gt[img_40_gt > max_value] = max_value
            img_41_gt[img_41_gt > max_value] = max_value
            img_42_gt[img_42_gt > max_value] = max_value
            img_43_gt[img_43_gt > max_value] = max_value

            img_00_LLT[img_00_LLT > max_value] = max_value
            img_01_LLT[img_01_LLT > max_value] = max_value
            img_02_LLT[img_02_LLT > max_value] = max_value
            img_03_LLT[img_03_LLT > max_value] = max_value



            #归一化
            img_00 = img_00 / max_value
            img_11 = img_11 / max_value
            img_22 = img_22 / max_value
            img_33 = img_33 / max_value

            img_40 = img_40 / max_value

            img_40_gt = img_40_gt / max_value
            img_41_gt = img_41_gt / max_value
            img_42_gt = img_42_gt / max_value
            img_43_gt = img_43_gt / max_value

            img_00_LLT = img_00_LLT / max_value
            img_01_LLT = img_01_LLT / max_value
            img_02_LLT = img_02_LLT / max_value
            img_03_LLT = img_03_LLT / max_value
            freq2 = np.full((600, 600), np.float(frequency2), dtype=np.float32)
            freq1 = np.full((600, 600), np.float(frequency1), dtype=np.float32)

            img_00, img_11, img_22, img_33, img_40, img_40_gt, img_41_gt, img_42_gt, img_43_gt, img_00_LLT, img_01_LLT, img_02_LLT, img_03_LLT, freq2, freq1 = \
                self.data_augmentation(img_00, img_11, img_22, img_33, img_40, img_40_gt, img_41_gt, img_42_gt, img_43_gt, img_00_LLT, img_01_LLT, img_02_LLT, img_03_LLT, freq2, freq1)
            
            freq2 = np.full((448, 448), np.float(frequency2), dtype=np.float32)
            freq1 = np.full((448, 448), np.float(frequency1), dtype=np.float32)

            img_00 = torch.from_numpy(img_00.copy()).unsqueeze(0)
            img_11 = torch.from_numpy(img_11.copy()).unsqueeze(0)
            img_22 = torch.from_numpy(img_22.copy()).unsqueeze(0)
            img_33 = torch.from_numpy(img_33.copy()).unsqueeze(0)
            img_40 = torch.from_numpy(img_40.copy()).unsqueeze(0)

            img_40_gt = torch.from_numpy(img_40_gt.copy()).unsqueeze(0)
            img_41_gt = torch.from_numpy(img_41_gt.copy()).unsqueeze(0)
            img_42_gt = torch.from_numpy(img_42_gt.copy()).unsqueeze(0)
            img_43_gt = torch.from_numpy(img_43_gt.copy()).unsqueeze(0)

            img_00_LLT = torch.from_numpy(img_00_LLT.copy()).unsqueeze(0)
            img_01_LLT = torch.from_numpy(img_01_LLT.copy()).unsqueeze(0)
            img_02_LLT = torch.from_numpy(img_02_LLT.copy()).unsqueeze(0)
            img_03_LLT = torch.from_numpy(img_03_LLT.copy()).unsqueeze(0)

            freq2 = torch.from_numpy(freq2.copy()).unsqueeze(0)
            freq1 = torch.from_numpy(freq1.copy()).unsqueeze(0)

            input_img = torch.cat((img_00, img_11, img_22, img_33, img_40_gt, img_41_gt,img_42_gt,img_43_gt, freq1, img_01_LLT, img_02_LLT, img_03_LLT ), 0)
            if i == 0:
                res_tensor = input_img
        return res_tensor





    def data_augmentation(self, img_00, img_11, img_22, img_33, img_40, img_40_gt, img_41_gt, img_42_gt, img_43_gt, img_00_LLT, img_01_LLT, img_02_LLT, img_03_LLT,  freq2, freq1):

        h, w = img_00.shape

        if self.dataset_name == 'train':
            x = np.random.randint(0, h - 448 + 1)
            y = np.random.randint(0, w - 448 + 1)
        else:
            x = 0
            y = 0

        img_00, img_11, img_22, img_33, img_40, img_40_gt, img_41_gt, img_42_gt, img_43_gt, img_00_LLT, img_01_LLT, img_02_LLT, img_03_LLT,  freq1, freq2 = \
            self.crop(img_00, img_11, img_22, img_33, img_40, img_40_gt, img_41_gt, img_42_gt, img_43_gt, img_00_LLT, img_01_LLT, img_02_LLT, img_03_LLT,  freq2, freq1, x,y,448, 448)

        if self.dataset_name == 'train':
            if random.uniform(0, 1) < 0.5:
                img_00 = img_00[:, ::-1]
                img_11 = img_11[:, ::-1]
                img_22 = img_22[:, ::-1]
                img_33 = img_33[:, ::-1]
                img_40 = img_40[:, ::-1]
                img_40_gt = img_40_gt[:, ::-1]
                img_41_gt = img_41_gt[:, ::-1]
                img_42_gt = img_42_gt[:, ::-1]
                img_43_gt = img_43_gt[:, ::-1]
                img_00_LLT = img_00_LLT[:, ::-1]
                img_01_LLT = img_01_LLT[:, ::-1]
                img_02_LLT = img_02_LLT[:, ::-1]
                img_03_LLT = img_03_LLT[:, ::-1]
            
            if random.uniform(0, 1) < 0.5:
                img_00 = img_00[::-1, :]
                img_11 = img_11[::-1, :]
                img_22 = img_22[::-1, :]
                img_33 = img_33[::-1, :]
                img_40 = img_40[::-1, :]
                img_40_gt = img_40_gt[::-1, :]
                img_41_gt = img_41_gt[::-1, :]
                img_42_gt = img_42_gt[::-1, :]
                img_43_gt = img_43_gt[::-1, :]
                img_00_LLT = img_00_LLT[::-1, :]
                img_01_LLT = img_01_LLT[::-1, :]
                img_02_LLT = img_02_LLT[::-1, :]
                img_03_LLT = img_03_LLT[::-1, :]
            
            p = random.uniform(0, 1)
            if p < 0.25:
                img_00 = cv2.rotate(img_00, cv2.ROTATE_90_CLOCKWISE)
                img_11 = cv2.rotate(img_11, cv2.ROTATE_90_CLOCKWISE)
                img_22 = cv2.rotate(img_22, cv2.ROTATE_90_CLOCKWISE)
                img_33 = cv2.rotate(img_33, cv2.ROTATE_90_CLOCKWISE)
                img_40 = cv2.rotate(img_40, cv2.ROTATE_90_CLOCKWISE)
                img_40_gt = cv2.rotate(img_40_gt, cv2.ROTATE_90_CLOCKWISE)
                img_41_gt = cv2.rotate(img_41_gt, cv2.ROTATE_90_CLOCKWISE)
                img_42_gt = cv2.rotate(img_42_gt, cv2.ROTATE_90_CLOCKWISE)
                img_43_gt = cv2.rotate(img_43_gt, cv2.ROTATE_90_CLOCKWISE)
                img_00_LLT = cv2.rotate(img_00_LLT, cv2.ROTATE_90_CLOCKWISE)
                img_01_LLT = cv2.rotate(img_01_LLT, cv2.ROTATE_90_CLOCKWISE)
                img_02_LLT = cv2.rotate(img_02_LLT, cv2.ROTATE_90_CLOCKWISE)
                img_03_LLT = cv2.rotate(img_03_LLT, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img_00 = cv2.rotate(img_00, cv2.ROTATE_180)
                img_11 = cv2.rotate(img_11, cv2.ROTATE_180)
                img_22 = cv2.rotate(img_22, cv2.ROTATE_180)
                img_33 = cv2.rotate(img_33, cv2.ROTATE_180)
                img_40 = cv2.rotate(img_40, cv2.ROTATE_180)
                img_40_gt = cv2.rotate(img_40_gt, cv2.ROTATE_180)
                img_41_gt = cv2.rotate(img_41_gt, cv2.ROTATE_180)
                img_42_gt = cv2.rotate(img_42_gt, cv2.ROTATE_180)
                img_43_gt = cv2.rotate(img_43_gt, cv2.ROTATE_180)
                img_00_LLT = cv2.rotate(img_00_LLT, cv2.ROTATE_180)
                img_01_LLT = cv2.rotate(img_01_LLT, cv2.ROTATE_180)
                img_02_LLT = cv2.rotate(img_02_LLT, cv2.ROTATE_180)
                img_03_LLT = cv2.rotate(img_03_LLT, cv2.ROTATE_180)
            elif p < 0.75:
                img_00 = cv2.rotate(img_00, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_11 = cv2.rotate(img_11, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_22 = cv2.rotate(img_22, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_33 = cv2.rotate(img_33, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_40 = cv2.rotate(img_40, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_40_gt = cv2.rotate(img_40_gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_41_gt = cv2.rotate(img_41_gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_42_gt = cv2.rotate(img_42_gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_43_gt = cv2.rotate(img_43_gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_00_LLT = cv2.rotate(img_00_LLT, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_01_LLT = cv2.rotate(img_01_LLT, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_02_LLT = cv2.rotate(img_02_LLT, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_03_LLT = cv2.rotate(img_03_LLT, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return img_00, img_11, img_22, img_33, img_40, img_40_gt, img_41_gt, img_42_gt, img_43_gt, img_00_LLT, img_01_LLT, img_02_LLT, img_03_LLT, freq2, freq1


                          
    def __getitem__(self, index):

        input_img = self.getimg(index)
        return input_img
