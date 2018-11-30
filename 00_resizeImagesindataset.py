from PIL import Image
import os
 
from os.path import basename


# main_with_ini.py
import configparser


 
def resizeImages(baseDir, training_images_base_width=558):
    basewidth = training_images_base_width
    for filename in os.listdir(baseDir):
        filenameOnly, file_extension = os.path.splitext(filename)
        # print (file_extension)
        if (file_extension in ['.jpg', '.png']):
            filepath = baseDir + os.sep + filename
            img = Image.open(filepath)
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.save(filepath)
            print (filenameOnly, "Done")
    print('Done')

if __name__ == '__main__':

    # Load paths and dataset names from config.ini file

    config = configparser.ConfigParser()
    config.read('settings-config.ini')

    dataset_name = config['DEFAULT']['dataset_name']
    data_root_dir= config['DEFAULT']['data_root_dir']

    training_images_base_width = int(config['DEFAULT']['training_images_base_width'])

    dataset_images_dir="{}/{}/images".format(data_root_dir, dataset_name)
    print(dataset_images_dir)
    resizeImages(dataset_images_dir, training_images_base_width)