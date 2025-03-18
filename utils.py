import json

# Save dictionary to txt file
def save_dict_to_txt(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)

# Load dictionary from txt file
def load_dict_from_txt(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data