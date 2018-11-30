
# read paths and values from settings-config.ini
dataset_name=$(./readconfig_ini_file.py settings-config.ini DEFAULT dataset_name)
caffessd_root_dir=$(./readconfig_ini_file.py settings-config.ini DEFAULT caffessd_root_dir)

data_root_dir=$(./readconfig_ini_file.py settings-config.ini DEFAULT data_root_dir)

echo "${dataset_name}    ${caffessd_root_dir} $data_root_dir"


cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )

#root_dir=$cur_dir/../..
#caffessd_root_dir=/home/inayat/new_retraining_mobilenet/caffe


cd ${caffessd_root_dir}


redo=1
#dataset_name="frozenElsaDataSet" 
#ataset_name="ucwinObjects" 
data_root_dir="$data_root_dir/$dataset_name"

mapfile="$data_root_dir/labelmap.prototxt"

echo "$data_root_dir"
echo "$mapfile"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=png --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python $caffessd_root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/structure/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
