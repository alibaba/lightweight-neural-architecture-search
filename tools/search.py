import argparse
from tinynas.searchers import build_searcher
from tinynas.utils.dict_action import DictAction 

def parse_args():
    parser = argparse.ArgumentParser(description='Search a network model')
    parser.add_argument('config', help='search config file path')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    kwargs = dict(cfg_file=args.config)
    if args.cfg_options is not None:
        kwargs['cfg_options'] = args.cfg_options
    searcher = build_searcher(default_args = kwargs)
    searcher.run()

if __name__ == '__main__':
    main()
