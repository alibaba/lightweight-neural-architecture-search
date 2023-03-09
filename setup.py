import re, os
import shutil
import setuptools

def get_install_requirements():
    with open('requirements/nas.txt', 'r', encoding='utf-8') as fin:
        reqs = [x.strip() for x in fin.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith('#')]
    return reqs

with open('tinynas/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(),
                        re.MULTILINE).group(1)

with open('README.md', encoding='utf-8') as f:
    readme_md = f.read()

def pack_resource():
    # pack resource such as configs and tools
    root_dir = os.path.dirname(__file__)
    proj_dir = os.path.join(root_dir,'tinynas')

    filenames = ['configs', 'tools']
    for filename in filenames:
        src_dir = os.path.join(root_dir, filename)
        dst_dir = os.path.join(proj_dir, filename)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

if __name__ == '__main__':
    pack_resource()
    setuptools.setup(
        name='tinynas',
        version=version,
        author='TinyML team of Alibaba DAMO Academy',
        description='A lightweight zero-short NAS toolbox for backbone search',
        long_description=readme_md,
        long_description_content_type='text/markdown',
        license='Apache License',
        packages=setuptools.find_packages(exclude=['*configs*', '*tools*', '*modelscope*']),
        include_package_data=True,
        install_requires=get_install_requirements(),
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent'
        ],
    )
