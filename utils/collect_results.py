import os, re
import pandas as pd
import json
from tqdm import tqdm

# Note: rename_subfolders is not used here.
# I keep it here for when I later complete the search_hparams.py script WITH data collection. (And tmpfs mounting, copying etc.)
def rename_subfolders(directory_path):
    """
    Rename trial result folders to shortened names.

    E.g.
    rename_subfolders("/home/jakob/ray_results3/LightningTrainer_2023-07-31_17-28-20")

    Names the runs:
    00000_0, 00000_1, 00000_2, ...
    """

    # Define the regular expression pattern to find the desired substring
    pattern = r'_\w+_(\d+_\d+)'

    # Iterate through the subdirectories in the given directory
    for subfolder in os.listdir(directory_path):
        subfolder_path = os.path.join(directory_path, subfolder)

        # Check if the item is a directory and not a file
        if os.path.isdir(subfolder_path):
            # Extract the desired substring from the subfolder name using the regular expression
            matches = re.findall(pattern, subfolder)
            if matches:
                new_name = matches[0]
                new_subfolder_path = os.path.join(directory_path, new_name)

                # Rename the subfolder
                os.rename(subfolder_path, new_subfolder_path)
                print(f'{subfolder_path} -> {new_subfolder_path}')

def read_lines(file_path):
    """
    Read a file and return a list of lines.
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def df_from_file(file_path):
    """
    Return a dataframe from a file that stores one json object per line.
    """

    lines = read_lines(file_path)
    dfs = [pd.read_json(line) for line in lines]
    df = pd.concat(dfs)

    new_column_names = {col: col.lstrip('_').replace('_', ' ').title() for col in df.columns}
    df.rename(columns=new_column_names, inplace=True)

    return df

def flatten_json(json_obj, parent_key='', sep='_'):
    """
    Flatten a nested json object.

    E.g.
    flatten_json({'a': 1, 'b': {'c': 2, 'd': {'e': 3}}})
    {'a': 1, 'b_c': 2, 'b_d_e': 3}
    """

    items = {}
    for key, value in json_obj.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_json(value, new_key, sep))
        else:
            items[new_key] = value
    return items

def read_params(file_path, name = ''):
    """
    Read and convert a ray tune params.json file to a dataframe.
    """

    # read file as string
    with open(file_path, 'r') as file:
        content = file.read()

    content = json.loads(content)
    content = flatten_json(content)

    df = pd.DataFrame.from_dict(content, orient='index')
    df = df.rename(columns={0: 'value'})
    df = df.sort_index()
    df = df.reset_index()
    df = df.rename(columns={'index': 'parameter'})

    str_to_replace = [
        'lightning_config_', 'model_config_', 'optimizer_config_', 'scheduler_config_', 'data_config_', 'augmentation_config_',
        'callbacks_config_', 'trainer_config_', 'tune_config_', 'scaling_config_', 'run_config_', 'checkpoint_config_', 'early_stopping_config_',
        'logging_config_','model_','optimizer_','scheduler_', 'data_', 'augmentation_', 'callbacks_', 'trainer_', '_model_',
        '_module_init_config_hparams_', '_init_config_', '_fit_params_'
    ]

    for str_ in str_to_replace:
        df['parameter'] = df['parameter'].str.replace(str_, '')

    df['parameter'] = df['parameter'].str.lstrip('_')
    df['parameter'] = df['parameter'].str.replace('_', ' ')
    df['parameter'] = df['parameter'].str.title()

    # sort by parameter
    df = df.sort_values(by=['parameter'])
    df.set_index('parameter', inplace=True)
    df = df.T

    # insert name column
    df.insert(0, 'Name', name)

    return df

def get_best_result(result_df):
    """
    Return the best result from a dataframe that stores the results of multiple runs.
    """

    # the first row stores the number of parameters
    number_of_params = int(result_df['Number Of Params'].values[0])
    number_of_trainable_params = int(result_df['Number Of Trainable Params'].values[0])

    best_result = result_df.sort_values('Val Loss', ascending=True).iloc[0]
    out_df = pd.DataFrame(best_result).T

    # overwrite the number of parameters
    out_df['Number Of Params'] = number_of_params
    out_df['Number Of Trainable Params'] = number_of_trainable_params

    return out_df

def get_run_result(folder_path):
    """
    Get the best result from a single trial.
    """

    result_path = os.path.join(folder_path, 'result.json')
    results = df_from_file(result_path)
    return get_best_result(results)

def transform_string(x):
    """
    This is a helper function that converts the trial id to the folder name, so
    params and results can be merged.
    """

    split_string = x.split('_')
    new_string = split_string[1] + '_' + str(int(split_string[1]))
    return new_string

def collect_data(file_path):
    """
    Collect the results and parameters from all subdirectories.
    Sort by best validation loss.
    """
    subdirs = [f.path for f in os.scandir(file_path) if f.is_dir() ]
    subdirs.sort()

    # get the results for all subdirectories
    print('Collecting results...')
    results = [get_run_result(folder_path) for folder_path in tqdm(subdirs, desc='Progress', unit='folder')]
    results = pd.concat(results, axis=0)
    results['Name'] = results['Trial Id'].apply(transform_string)

    # get the parameters for all subdirectories
    print('Collecting parameters...')
    params = [read_params(os.path.join(folder_path, 'params.json'), os.path.basename(folder_path)) for folder_path in tqdm(subdirs, desc='Progress', unit='folder')]
    params = pd.concat(params, axis=0)

    # merge results and parameters
    print('Merging results and parameters...')
    merged = pd.merge(params, results, on='Name')
    merged.sort_values(by='Val Loss', inplace=True)

    return merged


# Collect all results and parameters, sort by validation loss and output everything as csv.
file_path = '/home/jakob/ray_results3/LightningTrainer_2023-07-31_17-28-20' # folder with trials
out_path  = '/home/jakob/ray_results3/collected_data.csv'                   # output path

collected_data = collect_data(file_path)
print('Saving data...')
collected_data.to_csv(out_path, index=False)
print('Done.')
