cd "$(dirname "$0")"
set -e

cd ../../

image_size=224

flops="${1:-"41e8"}"
layers="${2:-"49"}"
name=madnas
work_dir=save_model/LightNAS/example/${name}

mkdir -p ${work_dir}

space_mutation="space_K1KXK1"
cp nas/spaces/${space_mutation}.py ${work_dir}
cp scripts/classification/example_madnas.sh ${work_dir}


echo "[\
{'class': 'ConvKXBNRELU', 'in': 3, 'out': 32, 's': 2, 'k': 3}, \
{'class': 'SuperResConvK1KXK1', 'in': 32, 'out': 128, 's': 2, 'k': 3, 'L': 3, 'btn': 32}, \
{'class': 'SuperResConvK1KXK1', 'in': 128, 'out': 256, 's': 2, 'k': 3, 'L': 4, 'btn': 64}, \
{'class': 'SuperResConvK1KXK1', 'in': 256, 'out': 512, 's': 2, 'k': 3, 'L': 6, 'btn': 128}, \
{'class': 'SuperResConvK1KXK1', 'in': 512, 'out': 1024, 's': 2, 'k': 3, 'L': 3, 'btn': 256}, \
]" \
  >${work_dir}/init_structure.txt

rm -rf acquired_gpu_list.*
mpirun --allow-run-as-root -np 1 -H 127.0.0.1:8 -bind-to none -map-by slot -mca pml ob1 \
  -mca btl ^openib -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  python nas/search.py configs/config_nas.py --work_dir ${work_dir} --cfg_options \
  task="classification" lat_pred=False only_master=True log_level="DEBUG" \
  budget_image_size=${image_size} budget_flops=${flops} budget_layers=${layers} \
  score_image_size=${image_size} score_multi_ratio=[0,0,0,0,1] score_type="madnas" \
  space_block_num=2 space_classfication=True space_mutation=${space_mutation} \
  space_structure_txt=${work_dir}/init_structure.txt
