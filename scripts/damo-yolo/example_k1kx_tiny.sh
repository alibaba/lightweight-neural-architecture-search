cd "$(dirname "$0")"
set -e

cd ../../

# 可调参数设定
image_size=320
image_channel=12

layers="${1:-"20"}"
latency="${2:-"8e-5"}"
last_ratio="${3:-"16"}"

# 工作空间准备
name=k1kx_R640__layers${layers}_lat${latency}_ratio${last_ratio}
work_dir=save_model/LightNAS/damo-yolo/${name}

mkdir -p ${work_dir}
space_mutation="space_K1KX"
cp nas/spaces/${space_mutation}.py ${work_dir}
cp scripts/damo-yolo/example_k1kx_small.sh ${work_dir}

echo "[\
{'class': 'ConvKXBNRELU', 'in': 12, 'out': 32, 's': 1, 'k': 3}, \
{'class': 'SuperResConvK1KX', 'in': 32, 'out': 48, 's': 2, 'k': 3, 'L': 1, 'btn': 32}, \
{'class': 'SuperResConvK1KX', 'in': 48, 'out': 96, 's': 2, 'k': 3, 'L': 1, 'btn': 48}, \
{'class': 'SuperResConvK1KX', 'in': 96, 'out': 96, 's': 2, 'k': 3, 'L': 1, 'btn': 96}, \
{'class': 'SuperResConvK1KX', 'in': 96, 'out': 192, 's': 1, 'k': 3, 'L': 1, 'btn': 96}, \
{'class': 'SuperResConvK1KX', 'in': 192, 'out': 384, 's': 2, 'k': 3, 'L': 1, 'btn': 192}, \
]" \
  >${work_dir}/init_structure.txt

rm -rf acquired_gpu_list.*
mpirun --allow-run-as-root -np 64 -H 127.0.0.1:64 -bind-to none -map-by slot -mca pml ob1 \
  -mca btl ^openib -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  python nas/search.py configs/config_nas.py --work_dir ${work_dir} \
  --cfg_options task="detection" space_classfication=False only_master=False log_level="INFO" \
  budget_image_size=${image_size} budget_flops=None budget_latency=${latency} budget_layers=${layers} budget_model_size=None \
  score_type="madnas" score_batch_size=32 score_image_size=${image_size} score_repeat=4 score_multi_ratio=[0,0,1,1,${last_ratio}] \
  budget_image_channel=${image_channel} score_image_channel=${image_channel} \
  lat_gpu=False lat_pred=True lat_date_type="FP16_DAMOYOLO" lat_pred_device="V100" lat_batch_size=32 \
  space_arch="MasterNet" space_mutation=${space_mutation} space_structure_txt=${work_dir}/init_structure.txt \
  ea_popu_size=512 ea_log_freq=5000 ea_num_random_nets=500000
