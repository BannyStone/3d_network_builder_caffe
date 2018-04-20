import os
from argparse import ArgumentParser
import pdb

parser = ArgumentParser(description=""" This script generates train & val prototxt files for action recognition""")
parser.add_argument('-m', '--main_branch', help="""normal, bottleneck""", required=True)

if __name__ == '__main__':
	args = parser.parse_args()
	pdb.set_trace()
