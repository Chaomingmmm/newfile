
#  import packages
import os
import zipfile

import cv2

def prepare_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    elif len(os.listdir(path))!= 0:
        for f in os.listdir(path):
            os.remove(os.path.join(path,f))

def resize_all_imgs(imgs_path,save_path = './resized_imgs',reshape_size = (64,64)):
    print(reshape_size)

    cnt = 0
    for image in os.listdir(imgs_path):
        print(image)
        img = cv2.imread(imgs_path+'/'+image)
    
        img = cv2.resize(img, reshape_size)
        cv2.imwrite(save_path+"/%d.png" % cnt,img)
        # print(img.shape)
        cnt += 1
    print('Successfully resized {cnt} images.\n'.format(cnt = str(cnt)))


if __name__ == "__main__":
    
    unzip_path = 'raw_imgs'
    prepare_path(unzip_path)

    resized_path = 'resized_imgs'
    prepare_path(resized_path)

    generated_path = 'generated_imgs'
    prepare_path(generated_path)

# unzip the dataset
    src_file = 'paintings1k.zip'
    with zipfile.ZipFile(os.path.join(src_file),"r") as zip_ref:
        zip_ref.extractall(unzip_path)
# resize all images and save into resized_imgs
    resize_all_imgs(unzip_path,resized_path,reshape_size=(32,32))

    print('done!')




