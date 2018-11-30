from PIL import Image
import os

import string

from os.path import basename
 
import configparser

def createTrainvalTxt(baseDirDataSet):
    buffer = ''
    baseDir = baseDirDataSet+'/images'
    for filename in os.listdir(baseDir):
        filenameOnly, file_extension = os.path.splitext(filename)
        # print (file_extension)
        s = 'images/'+filenameOnly+'.png'+' '+'labels/'+filenameOnly+'.xml\n'
        print (repr(s))
        img_file, anno = s.strip("\n").split(" ")
        #print(repr(img_file), repr(anno))
        buffer+=s
    with open(baseDirDataSet+'/structure/trainval.txt', 'w') as file:
        file.write(buffer)  
    
    print("{} is created successfully".format(baseDirDataSet+'/structure/trainval.txt'))   

    with open(baseDirDataSet+'/structure/test.txt', 'w') as file:
        file.write(buffer)  
    print("{} is created successfully".format(baseDirDataSet+'/structure/test.txt'))   
 

if __name__ == '__main__':


     # Load paths and dataset names from config.ini file

    config = configparser.ConfigParser()
    config.read('settings-config.ini')


    dataset_name = config['DEFAULT']['dataset_name']
    data_root_dir = config['DEFAULT']['data_root_dir']

    dataset_root_dir = "{}/{}".format(data_root_dir, dataset_name)

    createTrainvalTxt(dataset_root_dir)