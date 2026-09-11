import os
import pandas as pd
from pyfame.file_access.checks import *
from pathlib import Path
from importlib import resources

def make_paths(root_path:str = None, exclude_directories:list[str] | None = ["processed", "logs", "conversion", "analysis"]) -> pd.DataFrame:
    """ If the top level "data/" folder is not yet set up, this function will
    create the "data/" folder along with the necessary scaffolded subfolders, 
    and return a dataframe of absolute/relative file path pairs for all files
    within "data/". `exclude_directories` is used to configure which subdirectories
    are included in the OS walk of "data/", and by default it is configured to only
    return the file paths of the unmodified images/videos found in "data/raw/".

    Parameters
    ----------
    root_path : str
        An optional existing directory path where all of the projects
        data will be stored. By default this is None and this method will
        create a new "data/" folder at the project root.
    
    exclude_directories : list[str]
        A list of subfolder names to exclude from the returned dataframe
        of file paths. 

    Returns
    -------
    DataFrame

    Raises
    ------
    TypeError
        given invalid parameter typings.
    OSError
        given invalid file paths, or nonexistent directory names.
    
    """
    # The standard folder names for Pyfame input and output files. 
    dir_names = ["raw", "processed", "logs", "conversion", "analysis"]
    exclude_directories = set(exclude_directories or [])

    if root_path is not None:
        # If a root_path is provided, check that it is a valid path that exists in the CWD.
        check_type(root_path, [str])
        check_valid_path(root_path)
        check_is_dir(root_path)
    else:
        # If no root_path is provided, the data/ folder will become the root for all of Pyfame's i/o.
        root_path = os.path.join(os.getcwd(), "data")
        # Check if the directory path already exists; if not then create it. 
        if not os.path.isdir(root_path):
            os.mkdir(root_path)

    # For each directory name in dir_names, check if it already exists. 
    # If not, create the directory at the specified path.
    for dir in dir_names:
        path = os.path.join(root_path, dir)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print(f"Directory {dir} already exists within {root_path}.")
    
    print("-----------------------------------------------------------")
    
    full_file_paths = []
    rel_file_paths = []

    for path, dirs, files in os.walk(root_path, topdown=True):
        # Exclude unwanted subdirectories
        dirs[:] = [d for d in dirs if d not in exclude_directories]

        for file in files:
            full_path = os.path.join(path, file)
            rel_path = os.path.relpath(full_path, root_path)
            rel_path = os.path.join(os.path.basename(root_path), rel_path)
            
            full_file_paths.append(full_path)
            rel_file_paths.append(rel_path)

    df1 = pd.DataFrame({
        "Absolute Path":full_file_paths,
        "Relative Path":rel_file_paths,
    })

    return df1

def get_directory_walk(input_directory:str) -> pd.DataFrame:
        full_file_paths = []
        rel_file_paths = []

        for path, dirs, files in os.walk(input_directory, topdown=True):
            for file in files:
                full_path = os.path.join(path, file)
                rel_path = os.path.relpath(full_path, input_directory)
                rel_path = os.path.join(os.path.basename(input_directory), rel_path)
                
                full_file_paths.append(full_path)
                rel_file_paths.append(rel_path)

        df1 = pd.DataFrame({
            "Absolute Path":full_file_paths,
            "Relative Path":rel_file_paths,
        })

        return df1

def get_sub_directories_relative_to_path(file_path:str, anchor_directory:str) -> Path:
    path = Path(file_path).resolve()

    try:
        parts = path.parts
        anchor_idx = parts.index(anchor_directory)
        relative_parts = parts[anchor_idx + 1:-1]
        return Path(*relative_parts)
    except ValueError:
        raise ValueError(f"Anchor directory '{anchor_directory}' not found in path: {file_path}")

def get_landmarker_task_path() -> str:
    with resources.as_file(resources.files("pyfame.models.mediapipe") / "face_landmarker.task") as path:
        return str(path)
    
__all__ = ["make_paths", "get_directory_walk", "get_sub_directories_relative_to_path", "get_landmarker_task_path"]