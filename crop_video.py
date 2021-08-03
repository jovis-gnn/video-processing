import os
import sys
import pdb
import time
import pickle
import logging
import argparse
import subprocess
from glob import glob
from shutil import rmtree
from os.path import join, dirname, basename

import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d

from s3fd.inference import S3FD
from util import *


class CropFace:
    def __init__(self, config):
        """
        video_dirs should be set manually for different dataset.
        """
        self.single_video = config.single_video
        self.del_orig = config.del_orig
        self.ds_name = config.ds_name
        self.data_path = config.data_path
        self.video_ext = config.video_ext

        if not self.single_video:
            self.video_dirs = get_video_folder_for_dataset(self.ds_name, self.data_path)
        else:
            self.video_dirs = None
		
        self.out_path = config.out_path

        logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", datefmt="%I:%M:%S %p", level=logging.INFO)
        self.logger = logging.getLogger('preprocess_logger')

    def extract_av(self, video_path):
        """Extact frames in original video.
        1. It convert fps(25), sampling rate(16000) for video & audio respectively.
        2. Save frame images, video(converted), audio(converted) in 'orig' folder.
        3. video_idx and video_name should be set manually for different dataset.
        """

        if self.video_dirs is not None:
            out = get_out_folder_for_dataset(self.ds_name, video_path, self.out_path)
        else:
            out = self.out_path
	
        os.makedirs(join(out, 'orig'), exist_ok=True)

        video_out = join(out, 'orig', 'video.mp4')
        frame_out = join(out, 'orig', '%06d.jpg')
        wav_out = join(out, 'orig', 'audio.wav')

        command = ("ffmpeg -loglevel error -y -i {0} -qscale:v 2 -async 1 -r 25 {1}".format(video_path, video_out))
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel error -i {0} -qscale:v 2 -threads 1 -f image2 {1}".format(video_out, frame_out))
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel error -y -i {0} -ac 1 -vn -acodec pcm_s16le -ar 16000 {1}".format(video_out, wav_out))
        output = subprocess.call(command, shell=True, stdout=None)

        return out
	
    def face_detect(self, out):
        """Face detection with S3FD.
        1. It ignore detection for more than one identity per frame.
        2. Detection result format : [left, top, right, bottom, confidence]	
        """
        DET = S3FD(device='cuda')

        img_paths = glob(join(out, 'orig', '*.jpg'))
        img_paths.sort()

        dets = []

        for idx, img_path in enumerate(img_paths):
            image = cv2.imread(img_path)

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            DET.detect_faces(image_np, conf_th=0.9, scales=[0.25])
            try:
                bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[0.25])
            except:
                return None, 'BBox detection failed for {}\n'.format(img_path)

            if len(bboxes) >= 2:
                return None, 'More than one bbox in frame {}\n'.format(img_path)

            dets.append([])
            for bbox in bboxes:
                dets[-1].append({'frame': idx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})

        return len(img_paths), dets

    def bb_intersection_over_union(self, boxA, boxB):
        """Calculate ratio of intersection over union between two bboxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def track_shot(self, scenefaces):
        """
        1. If there are multiple identities in one frame, preprocess respectively.
        2. If some bboxes are missed, fill them by interpolation method.
        3. Check bbox scales whether it is affordable or not.
        """
        iouThres = 0.5
        tracks = []

        while True:
            track = []
            for framefaces in scenefaces:
                for face in framefaces:
                    if track == []:
                        track.append(face)
                        framefaces.remove(face)
                    elif face['frame'] - track[-1]['frame'] <= 25:
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        if iou > iouThres:
                            track.append(face)
                            framefaces.remove(face)
                            continue
                    else:
                        break

            if track == []:
                break
			
            framenum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])

            frame_i = np.arange(framenum[0], framenum[-1]+1)

            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)

            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > 100:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})
            else:
                return 'Mean bbox scale is too small\n'

        return tracks

    def crop_video(self, out, track):
        """Crop frames according to bbox.
        1. Face-tracking : Centering face in each frame using padding.
        """
        frames = glob(join(out, 'orig', '*.jpg'))
        frames.sort()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vOut = cv2.VideoWriter(join(out, 'tvideo.mp4'), fourcc, 25, (224, 224))

        dets = {'x': [], 'y': [], 's': []}
        for det in track['bbox']:
            dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
            dets['y'].append((det[1] + det[3]) / 2)
            dets['x'].append((det[0] + det[2]) / 2)
        dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
        dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

        for fidx, frame in enumerate(track['frame']):
            bbox = track['bbox'][fidx]
            cs = 0.4
            bs = dets['s'][fidx]
            bsi = int(bs * (1 + 2 * cs))

            image = cv2.imread(frames[frame])

            #Save cropped image for mouth generation
            #cropped_img = image[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
            #cv2.imwrite(join(out, '{}'.format(fidx).zfill(6) + '.jpg'), cropped_img)

            frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my = dets['y'][fidx] + bsi
            mx = dets['x'][fidx] + bsi

            face = frame[int(my-bs): int(my+bs*(1+2*cs)), int(mx-bs*(1+cs)): int(mx+bs*(1+cs))]

            vOut.write(cv2.resize(face, (224, 224)))

        audiotmp = join(out, 'audio.wav')
        audiostart = (track['frame'][0]) / 25
        audioend = (track['frame'][-1]+1) / 25
        
        vOut.release()

        # Crop audio file
        command = ("ffmpeg -loglevel error -y -i %s -ss %.3f -to %.3f %s" % (join(out, 'orig', 'audio.wav'), audiostart, audioend, audiotmp))
        output = subprocess.call(command, shell=True, stdout=None)
        if output != 0:
            pdb.set_trace()
        sample_rate, audio = wavfile.read(audiotmp)

        # Combine audio and video files
        command = ("ffmpeg -loglevel error -y -i %s -i %s %s" % (join(out, 'tvideo.mp4'), audiotmp, join(out, 'video.mp4')))
        output = subprocess.call([command], shell=True, stdout=None)
        if output != 0:
            pdb.set_trace()
        os.remove(join(out, 'tvideo.mp4'))


    def process_videos(self):
        self.logger.info('Start video preprocessing in {}'.format(self.data_path))
        processed = []
        n_processed = []
        for video in self.video_dirs:
            video_paths = sorted(glob(join(video, '*.{}'.format(self.video_ext))))
            for video_path in video_paths:
                self.logger.info('Now processing : {}'.format(video_path))

                out = self.extract_av(video_path)
                self.logger.info('Frames, Video(25), Audio(16000) saved.')

                frame_cnt, dets = self.face_detect(out)
                if isinstance(dets, str):
                    self.logger.warning(dets)
                    rmtree(out)
                    n_processed.append(out)
                    continue
                self.logger.info('Face detected : frame_cnt - {}, identity_cnt - {}'.format(frame_cnt, len(dets)))

                tracks = self.track_shot(dets)
                if isinstance(tracks, str):
                    self.logger.warning(tracks)
                    rmtree(out)
                    n_processed.append(out)
                    continue
                self.logger.info('Frame BBoxes are checked and interpolated.')

                for t in tracks:
                    self.crop_video(out, t)
                    if self.del_orig:
                        rmtree(join(out, 'orig'))
                processed.append(out)
                self.logger.info('Cropped video & audio generated successfully.\n')

        self.logger.info('Finished !')	
        self.logger.info('Processed video: {}, Skipped video: {}.\n\n'.format(len(processed), len(n_processed)))
		
        # Delete empty folder
        for p in n_processed:
            rmtree(p)

    def process_single_video(self):
        self.logger.info('Start video preprocessing for {}'.format(self.data_path))
		
        out = self.extract_av(self.data_path)
        self.logger.info('Frames, Video(25), Audio(16000) saved.')

        frame_cnt, dets = self.face_detect(out)
        if isinstance(dets, str):
            self.logger.warning(dets)
            rmtree(out)
            return
        self.logger.info('Face detected : frame_cnt - {}, identity_cnt - {}'.format(frame_cnt, len(dets)))

        tracks = self.track_shot(dets)
        if isinstance(tracks, str):
            self.logger.warning(tracks)
            rmtree(out)
            return
        self.logger.info('Frame BBox are checked and interpolated.')

        for t in tracks:
            self.crop_video(out, t)
            if self.del_orig:
                rmtree(join(out, 'orig'))
        self.logger.info('Cropped video & audio generated successfully.\n')

