import yaml

def read_yml(path):
    """
    Use to read yml files

    Args:
        path (str): Path of the file.

    Returns:
        dict: Content of the yml file
    """
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data
    