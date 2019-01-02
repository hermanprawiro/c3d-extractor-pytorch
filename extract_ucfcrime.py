import argparse
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from C3D_model import C3D
from dataset import FrameFolderDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='Path to input root folder')
parser.add_argument('--output_path', type=str, required=True, help='Path to output root folder')
parser.add_argument('--clip_stride', type=int, default=16)

def build_instances(features, output_file, num_instances=32):
    instances_start_ids = np.round(np.linspace(0, len(features) - 1, num_instances + 1)).astype(np.int)

    segments_features = []
    for i in range(num_instances):
        start = instances_start_ids[i]
        end = instances_start_ids[i + 1]

        if start == end:
            instance_features = features[start, :]
        elif end < start:
            instance_features = features[start, :]
        else:
            instance_features = torch.mean(features[start:end, :], dim=0)
        
        instance_features = torch.nn.functional.normalize(instance_features, p=2, dim=0)
        segments_features.append(instance_features.numpy())
    
    segments_features = np.array(segments_features)
    np.save(output_file, segments_features)

def save_raw(features, output_file):
    features = features.numpy()
    np.save(output_file, features)

@torch.no_grad()
def extract(model, input_path, output_file, clip_stride):
    print(input_path, output_file)
    images_data = FrameFolderDataset(input_path, clip_stride=clip_stride)
    dataloader = DataLoader(images_data, batch_size=16, shuffle=False, num_workers=4)
    
    video_output = []
    for sample in dataloader:
        sample = sample.to(device)
        print(sample.shape)
        output = model(sample)
        video_output.append(output.cpu())
    video_output = torch.cat(video_output, dim=0)

    save_raw(video_output, output_file)
    # build_instances(video_output, output_file)

def generate_train_test_list(input_path, output_path):
    TEMPORAL_ANNOTATION = os.path.join(input_path, 'Annotation', 'Temporal_Anomaly_Annotation.txt')
    temporal_list = [line.strip('\n') for line in open(TEMPORAL_ANNOTATION, 'r')]
    temporal_list = [row.split(' ')[0].replace('_x264.mp4', '') for row in temporal_list]
    
    train_list = []
    test_list = []
    for root, dirs, files in os.walk(os.path.join(input_path, 'Anomaly')):
        if len(files) > 0:
            # print('Processing', root)
            root_split = root.split('\\')
            if root_split[-1] in temporal_list:
                source_path = root
                output_folder = os.path.join(output_path, 'Test', *root_split[-3:-1])
                output_file = os.path.join(output_folder, root_split[-1] + '.npy')
                test_list.append((source_path, output_folder, output_file))
            else:
                source_path = root
                output_folder = os.path.join(output_path, 'Train', *root_split[-3:-1])
                output_file = os.path.join(output_folder, root_split[-1] + '.npy')
                train_list.append((source_path, output_folder, output_file))
    
    for root, dirs, files in os.walk(os.path.join(input_path, 'Normal')):
        if len(files) > 0:
            # print('Processing', root)
            root_split = root.split('\\')
            if root_split[-1] in temporal_list:
                source_path = root
                output_folder = os.path.join(output_path, 'Test', *root_split[-2:-1])
                output_file = os.path.join(output_folder, root_split[-1] + '.npy')
                test_list.append((source_path, output_folder, output_file))
            else:
                source_path = root
                output_folder = os.path.join(output_path, 'Train', *root_split[-2:-1])
                output_file = os.path.join(output_folder, root_split[-1] + '.npy')
                train_list.append((source_path, output_folder, output_file))
    
    return train_list, test_list

def main(input_path, output_path, clip_stride):
    model = C3D().to(device)
    model.load_state_dict(torch.load('c3d.pickle'))
    model.eval()

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

    train_list, test_list = generate_train_test_list(input_path, output_path)
    print('Done generating list')

    for row in train_list:
        source_path, output_folder, output_file = row
        print('Processing', source_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(output_file):
            extract(model, source_path, output_file, clip_stride)

    for row in test_list:
        source_path, output_folder, output_file = row
        print('Processing', source_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(output_file):
            extract(model, source_path, output_file, clip_stride)
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.clip_stride)