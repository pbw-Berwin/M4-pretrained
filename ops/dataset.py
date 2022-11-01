# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
from .video_utils import VideoClips
import torch
import random

class VideoRecord(object):
    def __init__(self, row, num_class):
        self._data = row[:2]
        self._data.append(torch.zeros(num_class))
        for cls in row[2:]:
            self._data[2][int(cls)] = 1.

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return self._data[2]

class VideoDataSetOnTheFly(data.Dataset):

    def __init__(self, root_path, list_file, num_groups=64, frames_per_group=1, sample_offset=0, num_clips=1,
                 modality='rgb', image_tmpl=None, dense_sampling=False, fixed_offset=True,
                 transform=None, is_train=True, test_mode=False, seperator=' ',
                 filter_video=0, num_classes=313, csail_format=False, metadir=None):
        """

        Argments have different meaning when dense_sampling is True:
            - num_groups ==> number of frames
            - frames_per_group ==> sample every K frame
            - sample_offset ==> number of clips used in validation or test mode

        Args:
            root_path (str): the file path to the root of video folder
            list_file (str): the file list, each line with folder_path, start_frame, end_frame, label_id
            num_groups (int): number of frames per data sample
            frames_per_group (int): number of frames within one group
            sample_offset (int): used in validation/test, the offset when sampling frames from a group
            modality (str): rgb or flow
            dense_sampling (bool): dense sampling in I3D
            fixed_offset (bool): used for generating the same videos used in TSM
            image_tmpl (str): template of image ids
            transform: the transformer for preprocessing
            is_train (bool): shuffle the video but keep the causality
            test_mode (bool): testing mode, no label
        """
        if modality not in ['rgb']:
            raise ValueError("modality should be 'flow' or 'rgb'.")

        self.root_path = root_path
        self.list_file = list_file
        self.num_groups = num_groups
        self.num_frames = num_groups
        self.frames_per_group = frames_per_group
        self.sample_freq = frames_per_group
        self.num_clips = num_clips
        self.sample_offset = sample_offset
        self.dense_sampling = dense_sampling
        self.modality = modality.lower()
        self.transform = transform
        self.is_train = is_train
        self.test_mode = test_mode
        self.seperator = seperator
        self.csail_format = csail_format
        self.num_consecutive_frames = 1
        self.fixed_offset = fixed_offset
        self.multi_label = None
        self._parse_list(num_classes)
        precomputed_metadata = self.read_metadata(metadir)
        self.video_clips = VideoClips(
            self.root_path,
            self.video_list,
            self.num_groups,
            self.frames_per_group,
            self.dense_sampling,
            30,
            _precomputed_metadata=precomputed_metadata,
            filter_video=filter_video,
            fixed_offset=fixed_offset,
            sample_clip_num=num_clips if not is_train and fixed_offset else None,
            num_workers=16,
        )
        self.label_list = self.video_clips.return_label
        if precomputed_metadata is None:
            self.save_metadata(metadir)
        self.num_classes = num_classes

    def read_metadata(self, metafile_dir):
        file_path = os.path.join(metafile_dir, '{}_metadata_{}.pth'.format(os.path.basename(metafile_dir), 'train' if self.is_train else 'val'))
        if os.path.isfile(file_path):
            metadata = torch.load(file_path)
            return metadata
        else:
            return None

    def save_metadata(self, metafile_dir):
        print('Saving meta data ...')
        torch.save(self.video_clips.metadata_full, \
            os.path.join(metafile_dir, '{}_metadata_{}.pth'.format(os.path.basename(metafile_dir), 'train' if self.is_train else 'val')))
        print('Finished')

    def _parse_list(self, num_class):
        # [video_id, class_idx]
        tmp = [x.strip().split(',') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 0]
        self.video_list = [VideoRecord(item, num_class) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def __getitem__(self, index):
        """
        Returns:
            torch.FloatTensor: (3xgxf)xHxW dimension, g is number of groups and f is the frames per group.
            torch.FloatTensor: the label
        """
        try:
            if self.is_train:
                video, video_idx = self.video_clips.get_clip(index)
                assert video_idx == index
                video_list = [Image.fromarray(img.numpy(), mode='RGB') for img in video]
            else:
                video_list = []
                for val_idx in range(self.num_clips):
                    if self.fixed_offset:
                        video, video_idx = self.video_clips.get_clip(index, val_idx)
                    else:
                        video, video_idx = self.video_clips.get_clip(index)
                    assert video_idx == index
                    video_list.extend([Image.fromarray(img.numpy(), mode='RGB') for img in video])
            images = self.transform(video_list)
            label = self.label_list[index]

            # re-order data to targeted format.
            return images, label
        except:
            return self.__getitem__(random.randrange(self.video_clips.num_videos()))
    
    def class_weights(self):
        weights = torch.zeros(self.num_classes)
        for rec in self.video_list:
          for label in rec.label.nonzero():
              weights[label] += 1
        weights = weights.clamp(1000,10000)
        #print(weights)
        weights = 1.0 / weights
        weights = weights.div(weights.max()).clamp(1e-5, 1)
        #print(weights)
        return weights


    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_videos()
