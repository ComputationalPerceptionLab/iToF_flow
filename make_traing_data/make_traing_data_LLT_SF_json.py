
import os
import cv2
import numpy as np
import imageio
import json
import math
DATA_PATH = '/xxxx/WFlowToF_data/'

seed = 1234
np.random.seed(seed)



mvoing_step = 1 
frequency = [20]

save_path = '/xxxx/ToF_AFLLT_SF_json/sequence'
train_txt = '/xxxx/ToF_AFLLT_SF_json/val_motion.txt'

train_path = '/xxxx/WFlowToF_data/val_motion.txt'
with open(train_path, 'r') as f:
    trainlist = f.read().splitlines()

step_gap = [1,2,3,4]
speed_ratios = [1] 


print(step_gap)

sence_cnt = 126
for sence in trainlist:
    sence_cnt += 1
    sence_path = os.path.join(save_path, '{0:05d}'.format(sence_cnt))
    

    if not os.path.exists(sence_path):
        os.makedirs(sence_path)

    
    data_cnt = 0
    step_gap = [1,2,3,4]
    for speed_ratio in speed_ratios:
        step_gap = [1,2,3,4]
        step_gap = [float(i* speed_ratio) for i in step_gap]
        step_gap = [int(round(i)) for i in step_gap]
        for i in range(step_gap[3], 50 - mvoing_step -1):
            for freq in frequency:
                for phase in range(4):
                    freq1 = freq
                    freq2 = freq
                    img_00_path_array = []
                    img_11_path_array = []
                    img_22_path_array = []
                    img_33_path_array = []
                    img_40_path_array = []

                    img_40_gt_path_array = []
                    img_41_gt_path_array = []
                    img_42_gt_path_array = []
                    img_43_gt_path_array = []

                    img_00_LLT_path_array = []
                    img_01_LLT_path_array = []
                    img_02_LLT_path_array = []
                    img_03_LLT_path_array = []
                    frequency2 = []

                    img_01_path_array = []
                    img_02_path_array = []
                    img_03_path_array = []

                    img_00_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[3], freq, phase)))
                    img_11_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[2], freq, (phase + 1) % 4)))
                    img_22_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[1], freq, (phase + 2) % 4)))
                    img_33_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[0], freq, (phase + 3) % 4)))

                    img_40_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i, freq, phase)))

                    img_40_gt_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i, freq, phase)))
                    img_41_gt_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i, freq, (phase + 1) % 4)))
                    img_42_gt_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i, freq, (phase + 2) % 4)))
                    img_43_gt_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i, freq, (phase + 3) % 4)))

                    img_00_LLT_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[3], freq, phase)))
                    img_01_LLT_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[2], freq, phase)))
                    img_02_LLT_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[1], freq, phase)))
                    img_03_LLT_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[0], freq, phase)))

                    img_01_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[3], freq, (phase + 1) % 4)))
                    img_02_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[3], freq, (phase + 2) % 4)))
                    img_03_path_array.append(os.path.join(DATA_PATH, sence, '{:03d}_render_{:d}MHz_phase{:d}.hdr'.format(i-step_gap[3], freq, (phase + 3) % 4)))




                    output_json = {
                        'img_00_path_array': img_00_path_array,
                        'img_11_path_array': img_11_path_array,
                        'img_22_path_array': img_22_path_array,
                        'img_33_path_array': img_33_path_array,

                        'img_40_path_array': img_40_path_array,

                        'img_40_gt_path_array': img_40_gt_path_array,
                        'img_41_gt_path_array': img_41_gt_path_array,
                        'img_42_gt_path_array': img_42_gt_path_array,
                        'img_43_gt_path_array': img_43_gt_path_array,

                        'img_00_LLT_path_array': img_00_LLT_path_array,
                        'img_01_LLT_path_array': img_01_LLT_path_array,
                        'img_02_LLT_path_array': img_02_LLT_path_array,
                        'img_03_LLT_path_array': img_03_LLT_path_array,
                        'freq': freq,
                        'img_01_path_array': img_01_path_array,
                        'img_02_path_array': img_02_path_array,
                        'img_03_path_array': img_03_path_array,
                    }
                    json_name = '{:03d}.json'.format(data_cnt)
                    with open(os.path.join(sence_path, json_name), 'w') as f:
                        json.dump(output_json, f)


                    write_sence_path = os.path.join( '{0:05d}'.format(sence_cnt))
                    with open(train_txt, 'a') as f:
                        f.write(os.path.join(write_sence_path, json_name) + '\n')
                    print('write {} to txt'.format(os.path.join(sence_path, json_name)))
                    data_cnt += 1

