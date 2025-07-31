from __future__ import division

import json
import logging
import open3d as o3d
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import os
from models.backbones.pointmae.pointmae import fps
from datasets.base_dataset import BaseDataset

logger = logging.getLogger("global_logger")


def build_custom_dataloader(cfg, training, distributed=True,is_cls=False):
    
    if(not is_cls):
        num_list = cfg["task_num"]
        str_n = ""
        str_list = []
        dataset_list = []
        data_loader_list = []
        for num in num_list:
            str_n += "_" + str(num)
        logger.info("building CustomDataset from: {}".format(cfg["meta_file"]+str_n+".json"))
        for id in range(len(num_list)):
            str_list.append(str_n + "_" + str(id+1)+".json")
            dataset = CustomDataset(
                cfg["meta_file"]+str_list[id],
                cfg["data_dir"],
                training,
            )
            dataset_list.append(dataset)

        for data_set in dataset_list:
            if(training):
                data_loader = DataLoader(
                    data_set,
                    batch_size=cfg["batch_size"],
                    num_workers=cfg["workers"],
                    pin_memory=True,
                    sampler=RandomSampler(data_set),
                )
            else:
                data_loader = DataLoader(
                    data_set,
                    batch_size=1,
                    num_workers=cfg["workers"],
                    pin_memory=True,
                    sampler=RandomSampler(data_set),
                )
            data_loader_list.append(data_loader)
    else:
        dataset = CustomDataset(
        cfg["meta_file"],
        cfg["data_dir"],
        training,
        )
        sampler = RandomSampler(dataset)
        if(training):
            data_loader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                num_workers=cfg["workers"],
                pin_memory=True,
                sampler=sampler,
                # worker_init_fn=worker_init_fn
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=cfg["workers"],
                pin_memory=True,
                sampler=sampler,
            )
        return data_loader
    return data_loader_list

def get_label_dict(directory):
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    subfolders = sorted(subfolders)
    # 创建字典存储子文件夹名称与第一个文件路径的映射
    subfolder_file_dict = {}
    
    for idx,subfolder in enumerate(subfolders):
        subfolder_file_dict[subfolder] = idx
    
    return subfolder_file_dict



class CustomDataset(BaseDataset):
    def __init__(
        self,
        meta_file,
        data_path,
        training,
    ):
        self.meta_file = meta_file
        self.training = training
        self.data_path = data_path
        if('Anomaly_shapeNet' in self.data_path):
            self.label_dict = get_label_dict(self.data_path)
        elif('MulSen_AD' in self.data_path):
            self.label_dict = get_label_dict(self.data_path)
        else:
            self.label_dict = {}
            subfolders = ["toffees","seahorse","fish","car","chicken","gemstone","starfish","candybar","duck","shell","airplane","diamond"]
            # subfolders = ["bowl4","cup0","bucket0","bottle0","tap1","headset1","vase3","helmet3","shelf0","cap0"]
            for idx,subfolder in enumerate(subfolders):
                self.label_dict[subfolder] = idx


        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)
                
    def norm_pcd(self, point_cloud):

        center = np.average(point_cloud,axis=0)
        # print(center.shape)
        new_points = point_cloud-np.expand_dims(center,axis=0)
        return new_points
        # centroid = np.mean(point_cloud, axis=0)
        # pc = point_cloud - centroid
        # m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        # pc = pc / m
        # return pc

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]
        test_path = self.data_path
        test_path = test_path.replace("real3D_down","Real3D-AD-PCD")
        filename = meta["filename"]
       

        if(self.training):
        # read image
            pcd = o3d.io.read_point_cloud(os.path.join(self.data_path,filename))
        else:
            pcd = o3d.io.read_point_cloud(os.path.join(test_path,filename))
        pointcloud = np.array(pcd.points)
        label = meta["label"]
        input.update(
            {
                "filename": filename,
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
            input["cls_label"] = self.label_dict[meta["clsname"]]
        else:
            input["clsname"] = filename.split("/")[-4]
            raise ValueError("Error! Dataset don't has clsname")


        # read / generate mask
        if meta.get("maskname", None):
            if('Anomaly_shapeNet' in self.data_path):
                pcd = np.genfromtxt(os.path.join(test_path,meta["maskname"]), delimiter=",")
            else:
                pcd = np.genfromtxt(os.path.join(test_path,meta["maskname"]), delimiter=" ")
            pointcloud = pcd[:,:3]
            mask = pcd[:,3]
        else:
            if label == 0:  # good
                mask = np.zeros((pointcloud.shape[0]))
            elif label == 1:  # defective
                mask = np.ones((pointcloud.shape[0]))
            else:
                raise ValueError("Labels must be [None, 0, 1]!")
            
        pointcloud = self.norm_pcd(pointcloud)


        pointcloud = transforms.ToTensor()(pointcloud)[0]
        # print(pointcloud.shape)
        point_num = pointcloud.shape[0]
        
        
        input.update({"pointcloud": pointcloud, "mask": mask,"point_num":point_num})
        return input
    
