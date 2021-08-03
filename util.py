from glob import glob
from os.path import join, dirname, basename 

def get_video_folder_for_dataset(ds_name, data_path):
    if ds_name == 'sentimental':
        video_dirs = sorted(glob(join(data_path, '*/movie')))
    elif ds_name == 'lrs':
        video_dirs = sorted(glob(join(data_path, 'main/*')))
    elif ds_name == 'vox' or ds_name == 'custom':
        video_dirs = sorted(glob(join(data_path, '*')))
    return video_dirs

def get_out_folder_for_dataset(ds_name, video_path, out_path):
    if ds_name == 'sentimental':
        video_name = basename(video_path)
        video_idx = video_name.split('-')[-1][:-5]
        video_name = '_'.join(video_name.split('-')[:-1])
        out = join(out_path, video_name, video_idx)
    elif ds_name == 'lrs':
        person_name = dirname(video_path).split('/')[-1]
        video_name = basename(video_path)[:-4]
        out = join(out_path, person_name, video_name)
    elif ds_name == 'vox' or ds_name == 'custom':
        video_name = dirname(video_path).split('/')[-1]
        video_idx = basename(video_path).split('_')[-1][:-4]
        out = join(out_path, video_name, video_idx)
    return out

