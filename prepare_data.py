import os
import random

import numpy as np

from shutil import copyfile

bacterial, normal, viral = [], [], []

# List all
for root, dirs, files in os.walk('raw/chest_xray'):
    for name in files:
        if name.endswith('.jpeg'):

            filename = os.path.join(root, name)
            
            xray_type = filename.split('/')[-2]

            if xray_type == 'NORMAL':
                normal.append(filename)
            elif xray_type != 'NORMAL':
                if 'virus' in filename:
                    viral.append(filename)
                elif 'bacteria' in filename:
                    bacterial.append(filename)

# EXPORT NORMAL DATA
random.shuffle(normal)
num_train, num_test, num_val = int(len(normal)*0.8), int(len(normal)*0.1), int(len(normal)*0.1)

for idx, normal_file in enumerate(normal[:num_train]):
    copyfile(normal_file, 'data/train/normal/normal_' + str(idx) + '.jpeg')

for idx, normal_file in enumerate(normal[num_train:num_train+num_test]):
    copyfile(normal_file, 'data/test/normal/normal_' + str(idx) + '.jpeg')

for idx, normal_file in enumerate(normal[num_train+num_test:]):
    copyfile(normal_file, 'data/validation/normal/normal_' + str(idx) + '.jpeg')

# EXPORT VIRAL DATA
random.shuffle(viral)
num_train, num_test, num_val = int(len(viral)*0.8), int(len(viral)*0.1), int(len(viral)*0.1)

for idx, viral_file in enumerate(viral[:num_train]):
    copyfile(viral_file, 'data/train/viral/viral_' + str(idx) + '.jpeg')

for idx, viral_file in enumerate(viral[num_train:num_train+num_test]):
    copyfile(viral_file, 'data/test/viral/viral_' + str(idx) + '.jpeg')

for idx, viral_file in enumerate(viral[num_train+num_test:]):
    copyfile(viral_file, 'data/validation/viral/viral_' + str(idx) + '.jpeg')

# EXPORT bacterial DATA
random.shuffle(bacterial)
num_train, num_test, num_val = int(len(bacterial)*0.8), int(len(bacterial)*0.1), int(len(bacterial)*0.1)

for idx, bacterial_file in enumerate(bacterial[:num_train]):
    copyfile(bacterial_file, 'data/train/bacterial/bacterial_' + str(idx) + '.jpeg')

for idx, bacterial_file in enumerate(bacterial[num_train:num_train+num_test]):
    copyfile(bacterial_file, 'data/test/bacterial/bacterial_' + str(idx) + '.jpeg')

for idx, bacterial_file in enumerate(bacterial[num_train+num_test:]):
    copyfile(bacterial_file, 'data/validation/bacterial/bacterial_' + str(idx) + '.jpeg')
    