import bisect
import math
from fractions import Fraction
from numpy import random
import torch
import numpy as np
from video_io import (
    _probe_video_from_file,
    _read_video_from_file,
    _read_video_timestamps_from_file,
    read_video,
    read_video_timestamps,
)

from tqdm import tqdm
import os


def pts_convert(pts, timebase_from, timebase_to, round_func=math.floor):
    """convert pts between different time bases
    Args:
        pts: presentation timestamp, float
        timebase_from: original timebase. Fraction
        timebase_to: new timebase. Fraction
        round_func: rounding function.
    """
    new_pts = Fraction(pts, 1) * timebase_from / timebase_to
    return round_func(new_pts)


def unfold(tensor, size, step, dilation=1):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors

    Returns all consecutive windows of `size` elements, with
    `step` between windows. The distance between each element
    in a window is given by `dilation`.
    """
    assert tensor.dim() == 1
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (step * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0] < 1:
        new_size = (0, size)
    return torch.as_strided(tensor, new_size, new_stride)


class _DummyDataset(object):
    """
    Dummy dataset used for DataLoader in VideoClips.
    Defined at top level so it can be pickled when forking.
    """

    def __init__(self, root, x):
        self.root = root
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return read_video_timestamps(os.path.join(self.root, self.x[idx]))


class VideoClips(object):
    """
    Given a list of video files, computes all consecutive subvideos of size
    `clip_length_in_frames`, where the distance between each subvideo in the
    same video is defined by `frames_between_clips`.
    If `frame_rate` is specified, it will also resample all the videos to have
    the same frame rate, and the clips will refer to this frame rate.

    Creating this instance the first time is time-consuming, as it needs to
    decode all the videos in `video_paths`. It is recommended that you
    cache the results after instantiation of the class.

    Recreating the clips for different clip lengths is fast, and can be done
    with the `compute_clips` method.

    Arguments:
        video_paths (List[str]): paths to the video files
        clip_length_in_frames (int): size of a clip in number of frames
        frames_between_clips (int): step (in frames) between each clip
        frame_rate (int, optional): if specified, it will resample the video
            so that it has `frame_rate`, and then the clips will be defined
            on the resampled video
        num_workers (int): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. (default: 0)
    """

    def __init__(
        self,
        root_path,
        video_list,
        num_groups=16,
        frames_per_group=1,
        dense_sampling=False,
        frame_rate=None,
        _precomputed_metadata=None,
        num_workers=0,
        fixed_offset=True,
        filter_video=30,
        sample_clip_num=None,
        frames_between_clips=1,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
        _video_max_dimension=0,
        _audio_samples=0,
        _audio_channels=0,
    ):
        self.root_path = root_path
        self.video_paths = [vid.path for vid in video_list]
        self.labels = [vid.label for vid in video_list]
        self.num_workers = num_workers

        # these options are not valid for pyav backend
        self._video_width = _video_width
        self._video_height = _video_height
        self._video_min_dimension = _video_min_dimension
        self._video_max_dimension = _video_max_dimension
        self._audio_samples = _audio_samples
        self._audio_channels = _audio_channels

        self.fixed_offset = fixed_offset
        self.is_train = True if sample_clip_num is None else False
        self.sample_clip_num = sample_clip_num
        self.filter_video = max(filter_video, frames_per_group*num_groups)
        self.frame_rate = frame_rate
        self.dense_sampling = dense_sampling
        self.frames_per_group = frames_per_group
        self.num_groups = num_groups
        self.num_frames = num_groups * frames_per_group if not self.dense_sampling else num_groups

        if _precomputed_metadata is None:
            self._compute_frame_pts()
        else:
            self._init_from_metadata(_precomputed_metadata)
        self.compute_clips(num_groups, frames_per_group, frames_between_clips, frame_rate)

    def _collate_fn(self, x):
        return x

    def _compute_frame_pts(self):
        self.video_pts = []
        self.video_fps = []

        # strategy: use a DataLoader to parallelize read_video_timestamps
        # so need to create a dummy dataset first
        import torch.utils.data

        dl = torch.utils.data.DataLoader(
            _DummyDataset(self.root_path, self.video_paths),
            batch_size=48,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

        with tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                clips, fps = list(zip(*batch))
                clips = [torch.as_tensor(c) for c in clips]
                self.video_pts.extend(clips)
                self.video_fps.extend(fps)
        
        self.metadata_full = {
            "video_paths": self.video_paths,
            "video_pts": self.video_pts,
            "video_fps": self.video_fps,
        }

        self.video_paths, self.video_pts, self.video_fps, self.label_list = self.filter_metadata(
            self.video_paths, self.video_pts, self.video_fps, self.labels
        )

    def _init_from_metadata(self, metadata):
        self.video_paths = metadata["video_paths"]
        assert len(self.video_paths) == len(metadata["video_pts"])
        self.video_pts = metadata["video_pts"]
        assert len(self.video_paths) == len(metadata["video_fps"])
        self.video_fps = metadata["video_fps"]

        self.metadata_full = {
            "video_paths": self.video_paths,
            "video_pts": self.video_pts,
            "video_fps": self.video_fps,
        }
        
        self.video_paths, self.video_pts, self.video_fps, self.label_list = self.filter_metadata(
            self.video_paths, self.video_pts, self.video_fps, self.labels
        )
        # self.video_paths=self.video_paths[6000:]
        # self.video_pts=self.video_pts[6000:]
        # self.video_fps=self.video_fps[6000:]
        # self.label_list=self.label_list[6000:]

    def filter_metadata(self, paths, pts, fps, labels):
        paths_new = []; pts_new = []; fps_new = []; label_new = [];
        for i, pt in enumerate(pts):
            if fps[i] is None: 
                fps[i] = 1
            total_frames = int(math.floor(len(pt) * float(self.frame_rate) / fps[i]))

            if total_frames >= self.filter_video:
                paths_new.append(paths[i])
                pts_new.append(pts[i])
                fps_new.append(fps[i])
                label_new.append(labels[i])
                # cat = paths[i].split('/')[-2]
                # label_new.append(self.action_idx[cat])
        assert len(paths_new) == len(pts_new)
        assert len(paths_new) == len(fps_new)
        assert len(paths_new) == len(label_new)
        print("The number of videos is {} (with more than {} frames) "
              "(original: {})".format(len(paths_new), self.filter_video, len(paths)), flush=True)
        return paths_new, pts_new, fps_new, label_new

    @property
    def metadata(self):
        _metadata = {
            "video_paths": self.video_paths,
            "video_pts": self.video_pts,
            "video_fps": self.video_fps,
        }
        return _metadata
        
    @property
    def return_label(self):
        return self.label_list

    def subset(self, indices):
        video_paths = [self.video_paths[i] for i in indices]
        video_pts = [self.video_pts[i] for i in indices]
        video_fps = [self.video_fps[i] for i in indices]
        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        return type(self)(
            video_paths,
            self.num_frames,
            self.step,
            self.frame_rate,
            _precomputed_metadata=metadata,
            num_workers=self.num_workers,
            _video_width=self._video_width,
            _video_height=self._video_height,
            _video_min_dimension=self._video_min_dimension,
            _video_max_dimension=self._video_max_dimension,
            _audio_samples=self._audio_samples,
            _audio_channels=self._audio_channels,
        )

    @staticmethod
    def compute_clips_for_video(video_pts, num_groups, frames_per_group, step, fps, frame_rate, dense_sampling):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps
        total_frames = len(video_pts) * (float(frame_rate) / fps)
        idxs = VideoClips._resample_video_idx(
            int(math.floor(total_frames)), fps, frame_rate
        )
        video_pts = video_pts[idxs]
        if dense_sampling:
            num_frames = num_groups * frames_per_group
        else:
            num_frames = max(1, len(video_pts))
        clips = unfold(video_pts, num_frames, step)
        if isinstance(idxs, slice):
            idxs = [idxs] * len(clips)
        else:
            idxs = unfold(idxs, num_frames, step)
        return clips, idxs

    def compute_clips(self, num_groups, frames_per_group, step, frame_rate=None):
        """
        Compute all consecutive sequences of clips from video_pts.
        Always returns clips of size `num_frames`, meaning that the
        last few frames in a video can potentially be dropped.

        Arguments:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
        """
        self.num_groups = num_groups
        self.frames_per_group = frames_per_group
        self.step = step
        self.frame_rate = frame_rate
        self.clips = []
        self.resampling_idxs = []
        for video_pts, fps in zip(self.video_pts, self.video_fps):
            clips, idxs = self.compute_clips_for_video(
                video_pts, num_groups, frames_per_group, step, fps, frame_rate, self.dense_sampling
            )
            self.clips.append(clips)
            self.resampling_idxs.append(idxs)
        clip_lengths = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]

    def get_clip_location(self, idx):
        """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def get_clip(self, idx, val_idx=None):
        """
        Gets a subclip from a list of videos.

        Arguments:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.num_clips())
            )
        start = self.cumulative_sizes[idx-1] if idx else 0
        end = self.cumulative_sizes[idx] - 1
        
        if self.is_train or not self.fixed_offset:
            global_clip_idx = random.choice(list(range(start, end+1)), 1)[0]
        else:
            if self.sample_clip_num == 1:
                global_clip_idx = (start + end) // 2
            else:
                assert val_idx < self.sample_clip_num
                global_clip_idx = (val_idx*end + (self.sample_clip_num-val_idx-1)*start) // (self.sample_clip_num-1)
        
        video_idx, clip_idx = self.get_clip_location(global_clip_idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        video, audio, info = read_video(os.path.join(self.root_path, video_path), start_pts, end_pts)
        
        if self.frame_rate is not None:
            resampling_idx = self.resampling_idxs[video_idx][clip_idx]
            if isinstance(resampling_idx, torch.Tensor):
                resampling_idx = resampling_idx - resampling_idx[0]
            video = video[resampling_idx]
            info["video_fps"] = self.frame_rate

        if self.dense_sampling:
            frame_idx = np.linspace(0, len(video) - 1, num=self.num_groups, dtype=int)
        else:
            max_frame_idx = len(video)
            if self.fixed_offset and not self.is_train:
                if self.sample_clip_num is None: self.sample_clip_num = 1
                if val_idx is None: val_idx=0
                sample_offsets = list(range(-self.sample_clip_num // 2 + 1, self.sample_clip_num // 2 + 1))
                sample_offset = sample_offsets[val_idx]
                if max_frame_idx > self.num_groups:
                    tick = max_frame_idx / float(self.num_groups)
                    curr_sample_offset = sample_offset
                    if curr_sample_offset >= tick / 2.0:
                        curr_sample_offset = tick / 2.0 - 1e-4
                    elif curr_sample_offset < -tick / 2.0:
                        curr_sample_offset = -tick / 2.0
                    frame_idx = np.array([int(tick / 2.0 + curr_sample_offset + tick * x) for x in range(self.num_groups)])
                else:
                    np.random.seed(sample_offset - (-self.num_clips // 2 + 1))
                    frame_idx = np.random.choice(max_frame_idx, self.num_groups)
            else:
                ave_frames_per_group = max_frame_idx // self.num_groups
                if ave_frames_per_group >= self.frames_per_group:
                    # randomly sample f images per segement
                    frame_idx = np.arange(0, self.num_groups) * ave_frames_per_group
                    frame_idx = np.repeat(frame_idx, repeats=self.frames_per_group)
                    offsets = np.random.choice(ave_frames_per_group, self.frames_per_group, replace=False)
                    offsets = np.tile(offsets, self.num_groups)
                    frame_idx = frame_idx + offsets
                elif max_frame_idx < total_frames:
                    # need to sample the same images
                    frame_idx = np.random.choice(max_frame_idx, total_frames)
                else:
                    # sample cross all images
                    frame_idx = np.random.choice(max_frame_idx, total_frames, replace=False)
            frame_idx = np.sort(frame_idx)
        # print(frame_idx)
        # print(clip_pts[frame_idx])
        video = video[frame_idx]
        assert len(video) == self.num_frames, "{} x {}".format(
            video.shape, self.num_frames
        )
        return video, video_idx