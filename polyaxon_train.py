# Polyaxon hyperparmeter tuning passes in parameters as command line arguments.
# Create a temporary config file based on args to pass to MUNIT trainer.
import argparse
import io
import os
import subprocess
import yaml
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('-p', action='append', nargs='+', help='Override parameter key and value')
opts = parser.parse_args()


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
    
def open_yaml(fn):
    with open(fn, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc
            
def write_yaml(data):
    ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    temp_fn = '/tmp/{}-{}.yaml'.format(os.environ['HOSTNAME'], ts)
    with io.open(temp_fn, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)
    return temp_fn

def parse(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
        
data = open_yaml(opts.config)
    
for (param, value) in opts.p:
    keys = param.split('.')
    nested_set(data, keys, parse(value))

temp_fn = write_yaml(data)

cmd = ['python', 'train.py', 
       '--config', temp_fn, 
       '--output_path', 'polyaxon',
       '--trainer', 'MUNIT']
print(cmd)
subprocess.run(cmd)
