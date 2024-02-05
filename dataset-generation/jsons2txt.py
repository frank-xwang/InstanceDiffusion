import os

path_to_json = 'train-data'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
with open('train.txt', 'w') as file:
    for l in json_files:
        file.write("{}/{}\n".format(path_to_json, l))
    file.close()
