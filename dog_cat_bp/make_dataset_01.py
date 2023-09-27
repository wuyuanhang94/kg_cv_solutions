import os, shutil

original_dataset_dir = '/home/yi/daily/red_peo_code/dog_cat/full'
base_dir = '/home/yi/daily/red_peo_code/dog_cat/small'
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
for dire in (train_dir, validation_dir, test_dir):
    if not os.path.isdir(dire):
        os.mkdir(dire)

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
for dire in (train_cats_dir, train_dogs_dir):
    if not os.path.isdir(dire):
        os.mkdir(dire)

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
for dire in (validation_cats_dir, validation_dogs_dir):
    if not os.path.isdir(dire):
        os.mkdir(dire)

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
for dire in (test_cats_dir, test_dogs_dir):
    if not os.path.isdir(dire):
        os.mkdir(dire)

fnames = [f'cat.{i}.jpg' for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f'cat.{i}.jpg' for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f'cat.{i}.jpg' for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)


fnames = [f'dog.{i}.jpg' for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f'dog.{i}.jpg' for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f'dog.{i}.jpg' for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

if __name__ == '__main__':
    print("total training cat images: ", len(os.listdir(train_cats_dir)))
    print("total training dog images: ", len(os.listdir(train_dogs_dir)))
    print("total validation cat images: ", len(os.listdir(validation_cats_dir)))
    print("total validation dog images: ", len(os.listdir(validation_dogs_dir)))
    print("total test cat images: ", len(os.listdir(test_cats_dir)))
    print("total test dog images: ", len(os.listdir(test_dogs_dir)))

# 平衡二分类问题