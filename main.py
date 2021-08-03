import logging
import argparse
from glob import glob
from os.path import join, dirname

from crop_video import CropFace
from offset import GetOffset
from synchronizer import synchronize

def sync_phase(out_path, opt, logger):
    logger.info('Getting offset value for {}'.format(out_path))
    os_getter = GetOffset(
        video_path=join(out_path, 'video.mp4')
    )
    os_getter.extract_data()
    offset, conf = os_getter.evaluate()
    logger.info('Before Offset : {}'.format(offset))

    logger.info('Synchronizing...')
    synchronize(
        video_path=join(out_path, 'video.mp4'),
        audio_path=join(out_path, 'audio.wav'),
        offset=offset
    )
    if opt.check_after_sync:
        os_getter.video_path = join(out_path, 'synced_video.mp4')
        os_getter.extract_data()
        offset, conf = os_getter.evaluate()
        logger.info('After Offset : {}'.format(offset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessor")
    parser.add_argument('--data_path', 
                        type=str, required=True, 
                        help='Input directory where videos are saved. For processing single video, specify video path.')
    parser.add_argument('--single_video', action='store_true')
    parser.add_argument('--ds_name', 
                        type=str, default='',
                        help='If you specify, check input directory and find videos according to dataset name.')
    parser.add_argument('--video_ext',
                        type=str, default='mp4',
                        help='extension of input video')
    parser.add_argument('--out_path',
                        type=str, default='',
                        help='Output directory. Default is same as data_path with suffix(_prep).')
    # For cropping
    parser.add_argument('--del_orig', action='store_true',
                        help='Delete original files or not')
    # For Synchronizing
    parser.add_argument('--check_after_sync', action='store_true')

    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", datefmt="%I:%M:%S %p", level=logging.INFO)
    logger = logging.getLogger('main_logger')

    opt = parser.parse_args()
    if opt.out_path == '' :
        if not opt.single_video:
            opt.out_path = opt.data_path + '_prep'
        else:
            opt.out_path = join(dirname(opt.data_path), 'prep')
    
    # Crop Video with Face Detection & Face Tracking
    logger.info('Cropping Phase.\n') 
    cropper = CropFace(config=opt)
    if opt.single_video:
        cropper.process_single_video()
    else:
        cropper.process_videos()

    # Get offset & synchronizing.
    logger.info('Synchronize Phase.\n')
    if opt.single_video:
        sync_phase(opt.out_path, opt, logger)
    else:
        process_paths = sorted(glob(join(opt.out_path, '*/*')))
        for p in process_paths:
            sync_phase(p, opt, logger)
    logger.info('Finished !')
        