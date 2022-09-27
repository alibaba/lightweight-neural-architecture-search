cd "$(dirname "$0")"
set -e

cd ../../

name=mbv2_flops300e6
work_dir=save_model/LightNAS/classification/${name}

mkdir -p ${work_dir}
space_mutation="space_k1dwk1"
cp nas/spaces/${space_mutation}.py ${work_dir}
cp scripts/classification/example_MBV2_FLOPs.sh ${work_dir}

echo "[\
{'class': 'ConvKXBNRELU', 'in': 3, 'out': 16, 's': 2, 'k': 3}, \
{'class': 'SuperResK1DWK1', 'in': 16, 'out': 24, 's': 2, 'k': 3, 'L': 1, 'btn': 48}, \
{'class': 'SuperResK1DWK1', 'in': 24, 'out': 48, 's': 2, 'k': 3, 'L': 1, 'btn': 96}, \
{'class': 'SuperResK1DWK1', 'in': 48, 'out': 64, 's': 2, 'k': 3, 'L': 1, 'btn': 128}, \
{'class': 'SuperResK1DWK1', 'in': 64, 'out': 96, 's': 1, 'k': 3, 'L': 1, 'btn': 192}, \
{'class': 'SuperResK1DWK1', 'in': 96, 'out': 192, 's': 2, 'k': 3, 'L': 1, 'btn': 384}, \
{'class': 'ConvKXBNRELU', 'in': 192, 'out': 1280, 's': 1, 'k': 1}, \
]" \
  >${work_dir}/init_structure.txt

rm -rf acquired_gpu_list.*
mpirun --allow-run-as-root -np 8 -H 127.0.0.1:8 -bind-to none -map-by slot -mca pml ob1 \
  -mca btl ^openib -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  python nas/search.py configs/config_nas.py --work_dir ${work_dir} \
  --cfg_options task="classification" space_classfication=True space_num_classes=1000 only_master=False log_level="INFO" \
  budget_image_size=224 budget_flops=300e6 budget_latency=None budget_layers=53 budget_model_size=None \
  score_type="entropy" score_batch_size=32 score_image_size=224 score_repeat=4 score_multi_ratio=[0,0,0,0,1] \
  space_arch="MasterNet" space_mutation=${space_mutation} space_structure_txt=${work_dir}/init_structure.txt
