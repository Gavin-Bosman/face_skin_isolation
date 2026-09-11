import os
import json
import pandas as pd
from datetime import datetime
from pyfame.file_access.checks import *
from pyfame.file_access.file_access_directories import create_output_directory

def analysis_to_disk(analysis_dictionary:dict[str, pd.DataFrame], analysis_label:str, working_directory_path:str = os.path.join(os.getcwd(), "data"),
                     output_folder_name:str | None = None) -> None:
    """ Given an analysis dictionary output containing (filename, DataFrame) pairs, 
    this function writes each analysis result to disk as a JSON file. Use this method
    to visualize results from `analyse_facial_colour_means`, `analyse_optical_flow_dense`
    or `analyse_optical_flow_sparse`.

    Parameters
    ----------
    analysis_dictionary : dict[str, DataFrame]
        a dictionary of (filename, DataFrame) pairs containing per-file
        analysis data.
    
    analysis_label : str
        A description of the analysis type performed used as metadata
        in the output JSON file.

    working_directory_path : str
        By default the "data/" folder in your current working directory;
        a path string to the folder where the analysis JSON files will be
        written.
    
    output_folder_name : str
        An optional subfolder name to organize this specific batch of
        analysis output files.
    
    Returns
    -------
    None

    Raises
    ------
    OSError
        Given invalid directory paths.
    
    """

    if not os.path.isdir(working_directory_path):
        raise OSError(
            message=f"Unable to locate the input {os.path.basename(working_directory_path)} directory."
            " Please call make_output_paths() to initialise the working directory."
        )

    # Get a unique folder identifier for this analysis session
    output_root = os.path.join(working_directory_path, "analysis")
    timestamp = datetime.now().isoformat(timespec='seconds')
    if not output_folder_name:
        output_folder_name = timestamp.replace(":","-")
    folder_path = create_output_directory(output_root, output_folder_name)

    for filename, df in analysis_dictionary.items():

        # Format the output file path
        file_path = os.path.join(folder_path, f"{filename}.json")
        analysis_dict = {
            f"{row['timestamp']:.5f}": row.drop('timestamp').to_dict()
            for _, row in df.iterrows()
        }

        # Format the output data
        output_dict = {
            "datetime":timestamp,
            "filename":filename,
            "analysis":analysis_label,
            "timestamp units": "seconds",
            "results":analysis_dict,
        }

        # Serialize to Json
        with open(file_path, "w") as f:
            json.dump(output_dict, f, indent=2)

__all__ = ["analysis_to_disk"]