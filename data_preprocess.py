import glob
import os
import cv2
import numpy as np
import shutil
import pandas as pd
import PIL.Image as Image
import sys
import getopt


# define functions
def crop_and_resize_images(dir_from_path, dir_to_path, width, height):
    # if dir_to_path does not exist, then create one; if exist, remove and recreate
    if (not os.path.exists(dir_to_path)):
        os.makedirs(dir_to_path)
    else:
        shutil.rmtree(dir_to_path)
        os.makedirs(dir_to_path)

    # crop images according to annotated ROI coordinates and resize to given dimensions
    print("Using: " + os.path.join(dir_from_path, '*.csv'))
    annotations = pd.read_csv(glob.glob(os.path.join(dir_from_path, '*.csv'))[0], sep=';')
    annotations = annotations.set_index('Filename')

    for image_name in os.listdir(dir_from_path):
        print("Processing Image: " + image_name)
        if image_name.endswith('.ppm'):
            x1 = annotations.at[image_name, 'Roi.X1']
            y1 = annotations.at[image_name, 'Roi.Y1']
            x2 = annotations.at[image_name, 'Roi.X2']
            y2 = annotations.at[image_name, 'Roi.Y2']
            image = Image.open(os.path.join(dir_from_path, image_name))
            image = image.crop((x1, y1, x2, y2))
            image = image.resize((width, height), resample=Image.BILINEAR)
            new_image_name = image_name.split(".")[-2] + ".png"
            print("Now image name is: " + new_image_name)
            image.save(os.path.join(dir_to_path, new_image_name))
        elif image_name.endswith('.csv'):
            file_from_path = os.path.join(dir_from_path, image_name)
            file_to_path = os.path.join(dir_to_path, image_name)
            shutil.copy(file_from_path, file_to_path)


def resize_images(dir_from_path, dir_to_path, width, height):
    # if dir_to_path does not exist, then create one; if exist, remove and recreate
    if (not os.path.exists(dir_to_path)):
        os.makedirs(dir_to_path)
    else:
        shutil.rmtree(dir_to_path)
        os.makedirs(dir_to_path)

    # resize to given dimensions
    for image_name in os.listdir(dir_from_path):
        print("Processing Image: " + image_name)
        if image_name.endswith('.png'):
            image = Image.open(os.path.join(dir_from_path, image_name))
            image = image.resize((width, height), resample=Image.BILINEAR)
            image.save(os.path.join(dir_to_path, image_name))


def equalize_normalize_images(path, width, height):
    for image_name in os.listdir(path):
        if image_name.endswith('.ppm') or image_name.endswith('.png'):
            img = cv2.imread(os.path.join(path, image_name), 0)

            # Gaussian filtering
            blur_img = cv2.GaussianBlur(img, (5, 5), 0)

            # equalize the histogram
            equalized_img = cv2.equalizeHist(blur_img)

            # normalize the image
            normalized_img = np.zeros((width, height))
            normalized_img = cv2.normalize(equalized_img, normalized_img, 0, 255, cv2.NORM_MINMAX)

            cv2.imwrite(os.path.join(path, image_name), normalized_img)


def read_data_with_roi(dir_from_path, dir_to_path, width, height):
    print(os.listdir(dir_from_path))
    for foldername in os.listdir(dir_from_path):
        print("Processing folder name is:" + foldername)
        if not (foldername.startswith('.')):
            crop_and_resize_images(os.path.join(dir_from_path, foldername),
                                   os.path.join(dir_to_path, foldername),
                                   width, height)
            equalize_normalize_images(os.path.join(dir_to_path, foldername), width, height)


def read_data_without_roi(dir_from_path, dir_to_path, width, height):
    for foldername in os.listdir(dir_from_path):
        print("Processing folder name is:" + foldername)
        if not (foldername.startswith('.')):
            resize_images(os.path.join(dir_from_path, foldername),
                          os.path.join(dir_to_path, foldername),
                          width, height)
            equalize_normalize_images(os.path.join(dir_to_path, foldername), width, height)


def main(argv):
    country = ''
    dir_from_path = ''
    dir_to_path = ''
    width = 0
    height = 0
    try:
        opts, args = getopt.getopt(argv,"c:i:o:w:h:",["country=","dir_from_path=", "dir_to_path=", "width=", "height="])
    except getopt.GetoptError:
        print("-c <country> -i <dir_from_path> -o <dir_to_path> -w <output_img_width> -h <output_img_height>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-c", "--country"):
            country = arg
        elif opt in ("-i", "--dir_from_path"):
            dir_from_path = arg
        elif opt in ("-o", "--dir_to_path"):
            dir_to_path = arg
        elif opt in ("-w", "--width"):
            width = int(arg)
        elif opt in ("-h", "--height"):
            height = int(arg)
        else:
            print("-c <country> -i <dir_from_path> -o <dir_to_path> -w <output_img_width> -h <output_img_height>")
            sys.exit(2)

    # to test width and height are integer
    try:
        int(width)
        int(height)
    except ValueError:
        print("Width and Height need to be integer")
        sys.exit(2)

    # determine which country's data
    if country == 'it':
        read_data_without_roi(dir_from_path, dir_to_path, int(width), int(height))
    elif country == 'ge' or country == 'be':
        read_data_with_roi(dir_from_path, dir_to_path, int(width), int(height))
    else:
        print("Country code are restricted to: it, ge, be")


if __name__ == "__main__":
    main(sys.argv[1:])

