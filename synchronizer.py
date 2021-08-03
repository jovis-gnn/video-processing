import os
import time
from os.path import join, dirname
import pickle
import argparse
import subprocess
from glob import glob
from shutil import rmtree

import cv2
from tqdm import tqdm
from scipy.io import wavfile

from offset import GetOffset


def synchronize(video_path, audio_path, offset, 
                tmp_dir='./tmp/sync', reference='synchronizer', center_offset=1):
    """
    Synchronize video & audio as to offset prediction.
    If offset is lower than standard offset(1), audio precedes video.
    Args:
        video_path: Target video folder path. It must contain audio file in same folder
        offset: Audio & video offset value by syncnet model. It means 0.04 seconds per unit.
            type: int
        save_path: Path to save synchronized video.
    """
    video_folder_path = dirname(video_path)
    save_video_path = join(video_folder_path, 'synced_video.mp4')
    save_audio_path = join(video_folder_path, 'synced_audio.wav')
    tmp_path = join(tmp_dir, reference)

    if os.path.exists(tmp_path):
        rmtree(tmp_path)
    os.makedirs(tmp_path)

    tmp_audio_path = join(tmp_path, 'audio.wav')
    tmp_video_path = join(tmp_path, 'video.mp4')

    video_frames = []
    vc = cv2.VideoCapture(video_path)
    while True:
        _, frame = vc.read()
        if frame is None:
            break
        video_frames.append(frame) 
    out_shape = video_frames[0].shape[:2]
    
    # Video Writer    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vOut = cv2.VideoWriter(tmp_video_path, fourcc, 25, out_shape)

    # Crop the front part of audio & video
    dist = (center_offset - offset)
    read_audio = audio_path
    if dist > 0:
        start = int(abs(dist)) * 0.04
        command = ("ffmpeg -loglevel error -y -i %s -ss %.3f %s" % (audio_path, start, tmp_audio_path))
        output = subprocess.call(command, shell=True, stdout=None)
        read_audio = tmp_audio_path
    else:
        video_start = int(abs(dist))
        video_frames = video_frames[video_start:]

    # Crop the back part of audio & video
    sample_rate, audio = wavfile.read(read_audio)
    min_duration = min((len(audio) / 16000), len(video_frames) / 25)
    video_end = int(min_duration*25)
    video_frames = video_frames[:video_end]

    # Make audio
    command = ("ffmpeg -loglevel error -y -i %s -ss %.3f -to %.3f %s" % (read_audio, 0, min_duration, save_audio_path))
    output = subprocess.call(command, shell=True, stdout=None)
    
    # Make video
    for f in video_frames:
        vOut.write(f)
    vOut.release()

    # Combine audio and video
    command = ("ffmpeg -loglevel error -y -i %s -i %s %s" % (tmp_video_path, save_audio_path, save_video_path))
    output = subprocess.call(command, shell=True, stdout=None)

    # Get video frames
    command = ("ffmpeg -loglevel error -y -i %s -threads 1 -f image2 %s" % (save_video_path, join(video_folder_path, '%06d.jpg')))
    output = subprocess.call(command, shell=True, stdout=None)
