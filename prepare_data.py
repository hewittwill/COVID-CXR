import os
import csv
import cv2
import random

from shutil import copyfile, rmtree

bacterial, normal, viral, covid = [], [], [], []

if not os.path.exists('raw/chest_xray') or not os.path.exists('raw/covid-chestxray-dataset'):
    print('RAW FILES NOT PRESENT - EXITING')
    exit()

rmtree('data/')

os.makedirs('data/train/normal')
os.makedirs('data/train/bacterial')
os.makedirs('data/train/viral')
os.makedirs('data/train/covid')

os.makedirs('data/test/normal')
os.makedirs('data/test/bacterial')
os.makedirs('data/test/viral')
os.makedirs('data/test/covid')

os.makedirs('data/validation/normal')
os.makedirs('data/validation/bacterial')
os.makedirs('data/validation/viral')
os.makedirs('data/validation/covid')

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
    im = cv2.imread(normal_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/train/normal/normal_' + str(idx) + '.jpeg', im)

for idx, normal_file in enumerate(normal[num_train:num_train+num_test]):
    im = cv2.imread(normal_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/test/normal/normal_' + str(idx) + '.jpeg', im)

for idx, normal_file in enumerate(normal[num_train+num_test:]):
    im = cv2.imread(normal_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/validation/normal/normal_' + str(idx) + '.jpeg', im)

# EXPORT VIRAL DATA
random.shuffle(viral)
num_train, num_test, num_val = int(len(viral)*0.8), int(len(viral)*0.1), int(len(viral)*0.1)

for idx, viral_file in enumerate(viral[:num_train]):
    im = cv2.imread(viral_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/train/viral/viral_' + str(idx) + '.jpeg', im)

for idx, viral_file in enumerate(viral[num_train:num_train+num_test]):
    im = cv2.imread(viral_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/test/viral/viral_' + str(idx) + '.jpeg', im)

for idx, viral_file in enumerate(viral[num_train+num_test:]):
    im = cv2.imread(viral_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/validation/viral/viral_' + str(idx) + '.jpeg', im)

# EXPORT bacterial DATA
random.shuffle(bacterial)
num_train, num_test, num_val = int(len(bacterial)*0.8), int(len(bacterial)*0.1), int(len(bacterial)*0.1)

for idx, bacterial_file in enumerate(bacterial[:num_train]):
    im = cv2.imread(bacterial_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/train/bacterial/bacterial_' + str(idx) + '.jpeg', im)

for idx, bacterial_file in enumerate(bacterial[num_train:num_train+num_test]):
    im = cv2.imread(bacterial_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/test/bacterial/bacterial_' + str(idx) + '.jpeg', im)

for idx, bacterial_file in enumerate(bacterial[num_train+num_test:]):
    im = cv2.imread(bacterial_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/validation/bacterial/bacterial_' + str(idx) + '.jpeg', im)    

# PREPARE COVID DATA
with open('raw/covid-chestxray-dataset/metadata.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for line in csv_reader:
        if 'PA' in line[6] and 'COVID' in line[4]:
            covid.append('raw/covid-chestxray-dataset/images/' + str(line[10]))
        
# EXPORT covid DATA
random.shuffle(covid)
num_train, num_test, num_val = int(len(covid)*0.8), int(len(covid)*0.1), int(len(covid)*0.1)

for idx, covid_file in enumerate(covid[:num_train]):
    im = cv2.imread(covid_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/train/covid/covid_' + str(idx) + '.jpeg', im)

for idx, covid_file in enumerate(covid[num_train:num_train+num_test]):
    im = cv2.imread(covid_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/test/covid/covid_' + str(idx) + '.jpeg', im)

for idx, covid_file in enumerate(covid[num_train+num_test:]):
    im = cv2.imread(covid_file)
    im = cv2.resize(im, (224, 224))
    cv2.imwrite('data/validation/covid/covid_' + str(idx) + '.jpeg', im)    