cd "$(dirname "$0")"
set -e

cd ../../

name=R50_R224_FLOPs41e8
work_dir=save_model/LightNAS/classification/${name}

mkdir -p ${work_dir}
space_mutation="space_K1KXK1"
cp nas/spaces/${space_mutation}.py ${work_dir}
cp scripts/classification/example_R50_FLOPs.sh ${work_dir}

echo "[\
{'class': 'ConvKXBNRELU', 'in': 3, 'out': 32, 's': 2, 'k': 3}, \
{'class': 'SuperResConvK1KXK1', 'in': 32, 'out': 256, 's': 2, 'k': 3, 'L': 1, 'btn': 64}, \
{'class': 'SuperResConvK1KXK1', 'in': 256, 'out': 512, 's': 2, 'k': 3, 'L': 1, 'btn': 128}, \
{'class': 'SuperResConvK1KXK1', 'in': 512, 'out': 768, 's': 2, 'k': 3, 'L': 1, 'btn': 256}, \
{'class': 'SuperResConvK1KXK1', 'in': 768, 'out': 1024, 's': 1, 'k': 3, 'L': 1, 'btn': 256}, \
{'class': 'SuperResConvK1KXK1', 'in': 1024, 'out': 2048, 's': 2, 'k': 3, 'L': 1, 'btn': 512}, \
]" \
  >${work_dir}/init_structure.txt

rm -rf acquired_gpu_list.*
mpirun --allow-run-as-root -np 8 -H 127.0.0.1:8 -bind-to none -map-by slot -mca pml ob1 \
  -mca btl ^openib -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  python nas/search.py configs/config_nas.py --work_dir ${work_dir} \
  --cfg_options task="classification" space_classfication=True space_num_classes=1000 only_master=False log_level="INFO" \
  budget_image_size=224 budget_flops=41e8 budget_latency=None budget_layers=49 budget_model_size=None \
  score_type="entropy" score_batch_size=32 score_image_size=224 score_repeat=4 score_multi_ratio=[0,0,0,0,1] \
  lat_gpu=False lat_pred=False lat_date_type="FP16" lat_pred_device="V100" lat_batch_size=32 \
  space_arch="MasterNet" space_mutation=${space_mutation} space_structure_txt=${work_dir}/init_structure.txt
