import setup_path
import yaml

file = open('config.yaml', 'r')
config = yaml.safe_load(file)
file.close()


folder_path = config['train_params']['folder_path']
print(len(folder_path))