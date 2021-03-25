import os
import uuid


def rename_folder(path):
    imgs = os.listdir(path)

    print('rename to uuid')
    for img in imgs:
        uuid_name = str(uuid.uuid4())
        os.rename(os.path.join(path, img), os.path.join(path, uuid_name + '.jpg'))
        print(img, '--->', uuid_name + '.jpg')

    print('rename by order')
    n = 0
    imgs = os.listdir(path)
    for img in imgs:
        os.rename(os.path.join(path, img), os.path.join(path, str(n) + '.jpg'))
        print(img, '--->', str(n) + '.jpg')
        n += 1


def rename_recurve(path):
    lst = os.listdir(path)
    dirs, files = [], []
    for l in lst:
        if os.path.isdir(os.path.join(path, l)):
            dirs.append(l)
        else:
            files.append(l)
    print(dirs, files)

    for dir in dirs:
        rename_recurve(os.path.join(path, dir))

    print('rename to uuid')
    imgs = []
    for img in files:
        uuid_name = str(uuid.uuid4())
        os.rename(os.path.join(path, img), os.path.join(path, uuid_name + '.jpg'))
        print(os.path.join(path, img), '--->', uuid_name + '.jpg')
        imgs.append(uuid_name + '.jpg')

    print('rename by order')
    n = 0
    for img in imgs:
        os.rename(os.path.join(path, img), os.path.join(path, str(n) + '.jpg'))
        print(os.path.join(path, img), '--->', str(n) + '.jpg')
        n += 1


if __name__ == '__main__':
    path = 'dataset/frs'
    rename_recurve(path)
