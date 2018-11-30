
# read paths and values from settings-config.ini
dataset_name=$(./readconfig_ini_file.py settings-config.ini DEFAULT dataset_name)

data_root_dir=$(./readconfig_ini_file.py settings-config.ini DEFAULT data_root_dir)

echo "${dataset_name}"


#dataset_name="frozenElsaDataSet" 
dataset_name="ucwinObjects" 

data_root_dir="$data_root_dir/$dataset_name"

echo "$data_root_dir"
	
#for trainval_lmdb
ln -s $data_root_dir/$dataset_name/lmdb/"$dataset_name"_trainval_lmdb/ trainval_lmdb 
 
#for test_lmdb
ln -s $data_root_dir/$dataset_name/lmdb/"$dataset_name"_test_lmdb/ test_lmdb 
 
#for labelmap.prototxt
 ln -s $data_root_dir/labelmap.prototxt labelmap.prototxt 
