import setup_path
import yaml

if __name__=="__main__":
    # Read YAML configurations
    try:
        file = open('config.yaml', 'r')
        config = yaml.safe_load(file)
        file.close()
    except Exception as error:
        print_msg(str(error), type=3)
        exit()

    initial_pose = eval(config['sim_params']['initial_pose'])
    if not isinstance(initial_pose[0], tuple):
        initial_pose = (initial_pose, initial_pose)
    
    print(initial_pose)

