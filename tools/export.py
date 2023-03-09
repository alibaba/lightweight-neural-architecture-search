import ast
import sys
from os.path import basename, expanduser, abspath, dirname, exists, isdir, join
from shutil import copy2, copytree, ignore_patterns, rmtree
import json

def export( work_dir, export_dir=None, force=False):

    taskid = basename(abspath(expanduser(work_dir))) 
    base_dir = dirname(dirname(abspath(__file__)))

    if export_dir is None:
        export_dir = '.'

    code_dir = join(base_dir, 'tinynas', 'deploy')

    # parse best
    best_from = join(work_dir, 'best_structure.txt')
    best_config = ast.literal_eval(open(best_from).read())
    arch = best_config['space_arch'].lower()

    # copy source code
    deploy_from = join(code_dir, arch)
    deploy_to = join(export_dir, taskid)
    if exists(deploy_to) and force:
        if isdir(deploy_to):
            rmtree(deploy_to)
        else:
            os.remove(deploy_to)
    copytree(
        deploy_from,
        deploy_to,
        ignore=ignore_patterns('__pycache__', '*.pyc', '*.md'))

    best_to = join(deploy_to, 'best_structure.json')
    json.dump(best_config, open(best_to, 'w'), indent=2)

    # copy weight
    weight_from = join(work_dir, 'weights')
    if exists(weight_from) and isdir(weight_from):
        weight_to = join(deploy_to, 'weights')
        copytree(weight_from, weight_to)

    return deploy_to


if __name__ == '__main__':
    args = {k: v for k, v in enumerate(sys.argv)}

    work_dir = args.get(1, None)
    output_dir = args.get(2, None)

    if work_dir is None:
        print('work_dir not specified!')
    else:
        loc = export(work_dir, output_dir, force=True)
        print('exported to', loc)
