import SimpleITK as sitk
import numpy as np
import os

from sklearn.model_selection import train_test_split, KFold
from glob import glob
import json

from monai.transforms import Resized, NormalizeIntensityd


def get_data_dic():
    data_dic = {
        'path': None,
        'wt': None,
        'tc': None,
        'et': None,
        'delta': None
    }
    return data_dic


def to_json(data, dest, json_name):
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, json_name), 'w') as f:
        json.dump(data, f, indent=4)
    

def split_dataset(root, dest, json_name):
    data_list = glob(root + '/*')

    image_train, image_test = train_test_split(data_list, test_size=0.2, random_state=32) 
    image_train, image_val = train_test_split(image_train, test_size=0.2, random_state=32)
    
    data = {
        "train":[], 
        "val": [],
        "test":[],
    }
    
    for i in image_train:
        image = []
        label = []
        temp = glob(i + '/*')
        for x in temp:
            if 'seg' not in x:
                image.append(x)
            else:
                label.append(x)
        

        data["train"].append({
            'image': image,
            'label': label
        })

    for i in image_test:
        image = []
        label = []
        temp = glob(i + '/*')
        for x in temp:
            if 'seg' not in x:
                image.append(x)
            else:
                label.append(x)
        
        data["test"].append({
            'image': image,
            'label': label
        })

    for i in image_val:
        image = []
        label = []
        temp = glob(i + '/*')
        for x in temp:
            if 'seg' not in x:
                image.append(x)
            else:
                label.append(x)
        
        data["val"].append({
            'image': image,
            'label': label
        })

    to_json(data, dest, json_name)
    
    return 0


def trans_brats_label(x):
        mask_WT = x.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 3] = 1

        mask_TC = x.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 3] = 1

        mask_ET = x.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 3] = 1
        
        mask = np.stack([mask_WT, mask_TC, mask_ET], axis=0)
        return mask


def get_box_and_size(x):
    """ 输入为胶质瘤切片，返回切片内三个标签的位置与尺寸
        input x: [155, 3, H, W] 
    """
    prompt = []
    
    for i in range(155):
        prompt_dic = {  
            'wt': None,
            'tc': None,
            'et': None
        }
        for j in range(3):
            if np.sum(x[i][j]) == 0:
                box_and_size = None
            else:
                loc = np.where(x[i][j] > 0)

                location = [int(np.min(loc[0])), int(np.min(loc[1])), 
                            int(np.max(loc[0])), int(np.max(loc[1]))]
                size = np.sum(x[i][j])

                location.append(int(size))
                box_and_size = location
                
            prompt_dic[list(prompt_dic.keys())[j]] = box_and_size

        prompt.append(prompt_dic)
    return prompt


def get_key_index(label):
    # 具有胶质瘤的开始与结束切片索引
    idx_start = -1
    idx_end = -1
    
    # tc 区域大于 9 的切片索引
    tc_start = -1
    tc_end = -1
    
    # 获取具有胶质瘤切片中 1/4， 2/4， 3/4， 的索引，用于编辑任务
    idx_1_quator = -1
    idx_mid = -1
    idx_3_quator = -1
    
    idx_list = []
    tc_list = []
    
    for i in range(0, 155):
        idx_list.append(np.sum(label[i, :, :, :]))
        tc_list.append(np.sum(label[i, 1, :, :]))

    # 包含胶质瘤的切片索引列表
    idx_list = np.where(np.array(idx_list) > 0)[0]
    idx_start = idx_list[0]
    idx_end = idx_list[-1]  
    
    # tc 区域大于 300 mm2 切片索引列表
    tc_list = np.where(np.array(tc_list) > 300)[0]
    
    if len(tc_list) < 6:
        return -1
    else:
        tc_start = min(tc_list)
        tc_end = max(tc_list)
        
        idx_mid = (tc_end + tc_start) // 2
        
        idx_1_quator = (idx_mid + tc_start) // 2
        idx_3_quator = (tc_end + idx_mid) // 2
        
        key_idx_list = [idx_start, tc_start, 
                        idx_1_quator, idx_mid, 
                        idx_3_quator, tc_end, 
                        idx_end]
        
        return key_idx_list


def get_mri_gen_json(prompt, case_dest, case_index, idx_list):
    json = []
    
    for idx in idx_list:
        data = get_data_dic()

        data['path'] = os.path.join(case_dest, "{}-{}.npz".format(case_index, idx))
        data['wt'] = prompt[idx]['wt']
        data['tc'] = prompt[idx]['tc']
        data['et'] = prompt[idx]['et']
        
        json.append(data)
    return json


def get_gli_gen_json(prompt, case_dest, case_index, idx_list):
    json = []
    
    for idx in idx_list:
        data = get_data_dic()
        
        data['path'] = os.path.join(case_dest, "{}-{}.npz".format(case_index, idx))
        data['wt'] = prompt[idx]['wt']
        data['tc'] = prompt[idx]['tc']
        data['et'] = prompt[idx]['et']
        
        json.append(data)
    return json


def get_gli_edit_json(prompt, label, case_dest, case_index, idx_list):
    if len(idx_list) != 6:
        return []
    
    json = []
    for i in range(5):
        data = get_gli_edit_case(prompt, label, case_dest, case_index, idx_list[i], idx_list[i + 1])
        json.append(data) 
        
    return json


def get_gli_edit_case(prompt, label, case_dest, case_index, start_idx, end_idx):
    data = get_data_dic()
    data['path'] = [os.path.join(case_dest, "{}-{}.npz".format(case_index, start_idx)),
                    os.path.join(case_dest, "{}-{}.npz".format(case_index, end_idx))]
    data['wt'] = [prompt[start_idx]['wt'], prompt[end_idx]['wt']] 
    data['tc'] = [prompt[start_idx]['tc'], prompt[end_idx]['tc']] 
    data['et'] = [prompt[start_idx]['et'], prompt[end_idx]['et']] 

    delta_wt = np.sum(np.bitwise_xor(label[start_idx][0], label[end_idx][0])) / np.sum(np.bitwise_or(label[start_idx][0], label[end_idx][0]))
    delta_tc = np.sum(np.bitwise_xor(label[start_idx][1], label[end_idx][1])) / np.sum(np.bitwise_or(label[start_idx][1], label[end_idx][1]))
    delta_et = np.sum(np.bitwise_xor(label[start_idx][2], label[end_idx][2])) / np.sum(np.bitwise_or(label[start_idx][2], label[end_idx][2]))
    data['delta'] = [delta_wt, delta_tc, delta_et]
    return data


def get_text_consis_case(prompt, case_dest, case_index, start_idx, mid_idx, end_idx):
    data = get_data_dic()

    data['wt'] = [prompt[start_idx]['wt'], prompt[mid_idx]['wt'], prompt[end_idx]['wt']] 
    data['tc'] = [prompt[start_idx]['tc'], prompt[mid_idx]['tc'], prompt[end_idx]['wt']] 
    data['et'] = [prompt[start_idx]['et'], prompt[mid_idx]['et'], prompt[end_idx]['wt']] 
    
    return data


def get_text_consis_json(prompt, case_dest, case_index, idx_list):
    if len(idx_list) < 6:
        return []
    
    json = []
    for i in range(5):
        data = get_text_consis_case(prompt, case_dest, case_index, 
                                 idx_list[i], 
                                 (idx_list[i] + idx_list[i + 1]) // 2, idx_list[i + 1])
        json.append(data) 
        
    return json


def get_data(path_list, dest, json_dest, phase='train'):    
    resize = Resized(keys=['image', 'label'], spatial_size=(155, 256, 256), mode='bilinear')
    norm = NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True)
    
    data = {
        'image': None,
        'label': None
    }
    
    mri_gen_json = []
    gli_gen_json = []
    gli_edit_json = []
    text_consis_json = []
    
    for i, path in enumerate(path_list):
        image_path_list = path['image']
        label_path = path['label'][0]
        
        case_index = os.path.split(label_path)[-1][:-11]
        
        case_dest = os.path.join(dest, case_index)
        
        os.makedirs(case_dest, exist_ok=True)
        
        image_list = []
        for image_path in sorted(image_path_list):
            image = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(image)     
            
            image_list.append(image)
            
        image = np.stack(image_list, axis=0)

        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)
        label = trans_brats_label(label)
        
        data['image'] = image.astype(np.float32)
        data['label'] = label.astype(np.int8)

        data = resize(data)
        data = norm(data)

        # [155, 4, 1, 256, 256]
        data['image'] = np.expand_dims(data['image'].get_array().transpose(1, 0, 2, 3), axis=2).astype(np.float32)
        
        # [155, 3, 256, 256]
        data['label'] = data['label'].get_array().transpose(1, 0, 2, 3).astype(np.int8)
        
        prompt = get_box_and_size(data['label'])
        key_idx = get_key_index(data['label'])
        
        if key_idx == -1:
            continue
        else:
            idx_start, tc_start, idx_1_quator, idx_mid, idx_3_quator, tc_end, idx_end = key_idx
            
            # Save multiple consecutive image and label pairs in the same case
            idx_text_consis_save = []
            for j in range(tc_start, tc_end, max(1, (tc_end - tc_start) // 6)):
                idx_text_consis_save += [j]
            idx_text_consis_save = sorted(set(idx_text_consis_save))[:6]
            print("Slice index used for gli edit and text regularization:", idx_text_consis_save)
            
            # Save selected image and label pairs
            idx_gli_image_save = []
            for j in range(tc_start, tc_end, min(1, (tc_end - tc_start) // 10)):
                idx_gli_image_save += [j]
            print("Slice index used for gli gen and image gen:", idx_gli_image_save)

            # Save all image and label pairs
            idx_imputation_save = []
            for j in range(tc_start, tc_end):
                idx_imputation_save += [j]
            idx_imputation_save = sorted(set(idx_imputation_save))
            print("Slice index used for gli gen and image gen:", idx_imputation_save)
            
            idx_used = idx_text_consis_save + idx_gli_image_save + idx_imputation_save
            
            seg_name = os.path.split(label_path)[-1][:-7]
            t1n_name = seg_name.replace('seg', 't1n')
            t1c_name = seg_name.replace('seg', 't1c')
            t2w_name = seg_name.replace('seg', 't2w')
            t2f_name = seg_name.replace('seg', 't2f')
            
            print("Save {} / {} image {}".format(i + 1, len(path_list), case_index))
            for idx in idx_used:
                if not os.path.exists(os.path.join(case_dest, "{}-{}.npz".format(case_index, idx))):
                    np.savez(os.path.join(case_dest, "{}-{}.npz".format(case_index, idx)), 
                            label=data['label'][idx].astype(np.int8), 
                            t1c=data['image'][idx][0], t1n=data['image'][idx][1],
                            t2f=data['image'][idx][2], t2w=data['image'][idx][3])

            # 获取 MRI 生成 json
            tmp = get_mri_gen_json(prompt, case_dest, case_index, idx_imputation_save)
            imputation_json += (tmp)
               
            # 获取 MRI 生成 json
            tmp = get_mri_gen_json(prompt, case_dest, case_index, idx_gli_image_save)
            mri_gen_json += (tmp)
            
            # 获取 gli 生成 json
            tmp = get_gli_gen_json(prompt, case_dest, case_index, idx_gli_image_save)
            gli_gen_json += (tmp)
            
            # 获取 gli 编辑 json
            tmp =  get_gli_edit_json(prompt, data['label'], case_dest, case_index, idx_text_consis_save)
            gli_edit_json += (tmp)
            
            # 获取文本正则化 json
            tmp = get_text_consis_json(prompt, case_dest, case_index, idx_text_consis_save)
            text_consis_json += (tmp)

    to_json(mri_gen_json, json_dest, "mri_gen_{}.json".format(phase))
    to_json(imputation_json, json_dest, "modality_imputation_{}.json".format(phase))
    to_json(gli_gen_json, json_dest, "gli_gen_{}.json".format(phase))
    to_json(gli_edit_json, json_dest, "gli_edit_{}.json".format(phase))
    to_json(text_consis_json, json_dest, "text_consis_{}.json".format(phase))

    return 0

def gather_json():
    train = None
    test = None
    val = None
    
    data = {
        'train': None,
        'val': None,
        'test': None
    }

    with open('./modality_imputation_train.json', 'r') as f:
        train = json.load(f)
        
    with open('./modality_imputation_val.json', 'r') as f:
        val = json.load(f)
        
    with open('./modality_imputation_test.json', 'r') as f:
        test = json.load(f)
        
    data['train'] = train
    data['val'] = val
    data['test'] = test

    with open('./modality_imputation.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    with open('./modality_imputation.json', 'r') as f:
        train = json.load(f)
        
    print(train.keys())
    print(len(train['train']), len(train['val']), len(train['test']))

    
def main(data_root=None, data_dest=None, json_dest=None, json_name=None):
    data_root = './Brats2023/Adult_Glioma/TrainingData'
    data_dest = './Brats2023/ag'
    json_dest = './Brats2023/ag'
    json_name = 'ag_train_val_test.json'
    
    split_dataset(data_root, dest=json_dest, json_name=json_name)
    
    with open(os.path.join(json_dest, json_name), 'r') as f:
        data = json.load(f)
    
    get_data(data['train'], os.path.join(data_dest, 'train'), json_dest, 'train')
    get_data(data['val'], os.path.join(data_dest, 'val'), json_dest, 'val')
    get_data(data['test'], os.path.join(data_dest, 'test'), json_dest, 'test')
    
    gather_json()

    return 0


if __name__ == "__main__":
    main()
    gather_json()


    

    
