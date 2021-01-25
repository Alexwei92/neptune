import os, sys

# Path settings
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
import_path = os.path.join(parent_dir, '.')
sys.path.insert(0, import_path)