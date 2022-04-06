# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os,sys
from posixpath import dirname
import ast, argparse, copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import save_pyobj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--suffix', type=str, default="New")

    args = parser.parse_args()
    return args


def load_pyobj(filename):
    with open(filename, 'r') as fid:
        the_s = fid.readlines()

    if isinstance(the_s, list):
        the_s = ''.join(the_s)

    the_s = the_s.replace('inf', '1e20')
    pyobj = ast.literal_eval(the_s)
    return pyobj


def write_structure_info(structure_info, structure_txt, suffix):
    new_structure_txt = structure_txt.replace(".txt", "_%s.txt"%(suffix))
    new_structure_info = copy.deepcopy(structure_info)
    for idx, block_info in enumerate(new_structure_info):
        if "SuperRes" in block_info["class"] and "inner_class" in block_info:
            block_info["class"] = block_info["class"]+"_%s"%(suffix)
            block_info["inner_class"] = block_info["inner_class"]+"_%s"%(suffix)
    save_pyobj(new_structure_txt, new_structure_info)


def main():
    args = parse_args()
    dict_info = load_pyobj(args.filename)
    structure_info = dict_info["popu_structure_list"][args.idx]
    print(structure_info)
    structure_txt = os.path.join(os.path.dirname(args.filename), "best_structure.txt")
    save_pyobj(structure_txt, structure_info)

    # write_structure_info(structure_info, structure_txt, args.suffix)
            

if __name__ == '__main__':
    main()