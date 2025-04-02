# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import os
import torch
from mmcv import Config

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='determine consequutive frames in nuscenes')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--train', help='scan the train set', default=False, action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # build the dataloader
    if args.train:
        dataset = build_dataset(cfg.data.train)
    else:
        dataset = build_dataset(cfg.data.test)
    # from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
    # dataset = CustomNuScenesDataset(**cfg.data.test.copy())
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )


    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility


    prev_scene_token = None
    dataset = data_loader.dataset
    scene_start = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if args.train:
                img_metas = data["img_metas"].data[0][0]
            else:
                img_metas = data["img_metas"][0].data[0]
            if img_metas[0]["scene_token"] != prev_scene_token:
                scene_start.append(i)
            prev_scene_token = img_metas[0]["scene_token"]
    scene_start.append(len(dataset))
    scene_starts_file = 'scene_starts'
    scene_lengths_file = 'scene_lengths'
    if args.train:
        scene_starts_file = 'train_' + scene_starts_file
        scene_lengths_file = 'train_' + scene_lengths_file
    with open('data/' + scene_starts_file + '.pkl', 'wb') as f:
        pickle.dump(scene_start, f)
    with open('data/' + scene_starts_file + '.txt', 'w') as file:
        file.write('\n'.join([str(x) for x in scene_start]))
    scene_lengths = []
    for i in range(len(scene_start)-1):
        length = scene_start[i+1]-scene_start[i]
        scene_lengths.append(length)
    with open('data/' + scene_lengths_file + '.pkl', 'wb') as f:
        pickle.dump(scene_lengths, f)
    with open('data/' + scene_lengths_file + '.txt', 'w') as file:
        file.write('\n'.join([str(x) for x in scene_lengths]))
if __name__ == '__main__':
    main()