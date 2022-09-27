cd "$(dirname "$0")"
set -e

cd ../../

image_size=224

flops="${1:-"300e6"}"
layers="${2:-"47"}"
init_bit="${3:-"4"}"
init_std="${4:-"4"}"
init_std_act="${5:-"5"}"
switch="${6:-"1"}"
name=mixed_${init_bit}bits_std${init_std_act}_${init_std}_flops${flops}_layers${layers}_btn116
work_dir=save_model/LightNAS/quant/${name}
if [ ${switch} = "1" ]; # train or test 
    then
  mkdir -p ${work_dir}

  space_mutation="space_quant_k1dwk1"
  cp nas/spaces/${space_mutation}.py ${work_dir}
  cp scripts/quant/mixed_19d2G.sh ${work_dir}

  bits_list="${init_bit},${init_bit},${init_bit}"
  bits_list2="$bits_list,$bits_list"
  bits_list3="$bits_list,$bits_list,$bits_list"
  bits_list4="$bits_list2,$bits_list2"
  bits_list5="$bits_list2,$bits_list3"
  bits_list6="$bits_list3,$bits_list3"

  echo "[\
  {'class': 'ConvKXBNRELU', 'in': 3, 'out': 16, 's': 2, 'k': 3, 'nbitsA':8, 'nbitsW':8}, \
  {'class': 'SuperResK1DWK1', 'in': 16, 'out': 24, 's': 2, 'k': 3, 'L': 1, 'btn': 48, 'nbitsA':[${bits_list}], 'nbitsW':[${bits_list}]}, \
  {'class': 'SuperResK1DWK1', 'in': 24, 'out': 48, 's': 2, 'k': 3, 'L': 1, 'btn': 96, 'nbitsA':[${bits_list}], 'nbitsW':[${bits_list}]}, \
  {'class': 'SuperResK1DWK1', 'in': 48, 'out': 64, 's': 2, 'k': 3, 'L': 1, 'btn': 128, 'nbitsA':[${bits_list}], 'nbitsW':[${bits_list}]}, \
  {'class': 'SuperResK1DWK1', 'in': 64, 'out': 96, 's': 1, 'k': 3, 'L': 1, 'btn': 192, 'nbitsA':[${bits_list}], 'nbitsW':[${bits_list}]}, \
  {'class': 'SuperResK1DWK1', 'in': 96, 'out': 192, 's': 2, 'k': 3, 'L': 1, 'btn': 384, 'nbitsA':[${bits_list}], 'nbitsW':[${bits_list}]}, \
  {'class': 'ConvKXBNRELU', 'in': 192, 'out': 1280, 's': 1, 'k': 1, 'nbitsA':${init_bit}, 'nbitsW':${init_bit}}, \
  ]" \
    >${work_dir}/init_structure.txt

  rm -rf acquired_gpu_list.*
  mpirun --allow-run-as-root -np 64 -H 127.0.0.1:8 -bind-to none -map-by slot -mca pml ob1 \
    -mca btl ^openib -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --oversubscribe \
    python nas/search.py configs/config_nas.py --work_dir ${work_dir} --cfg_options \
    task="classification" lat_pred=False only_master=False log_level="info" \
    budget_image_size=${image_size} budget_flops=${flops} budget_layers=${layers} \
    score_image_size=${image_size} score_multi_ratio=[0,0,1,1,6] score_quant_search=True \
    score_repeat=4 score_init_std=${init_std} score_init_std_act=${init_std_act} score_type="madnas" \
    space_block_num=2 space_classfication=True space_mutation=${space_mutation} \
    space_structure_txt=${work_dir}/init_structure.txt \
    ea_popu_size=512 ea_log_freq=5000 ea_num_random_nets=500000

    else
    pass
    
fi
