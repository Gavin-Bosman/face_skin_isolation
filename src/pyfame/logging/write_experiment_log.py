import json
import os
from enum import Enum
from datetime import datetime
from importlib.resources import files
import jsonschema
from jsonschema import ValidationError
from pyfame.layer._layer import Layer
from pyfame.utils.general_utilities import get_landmark_names

def make_json_serializable(obj):
    """
    Recursively converts an object into json serializable types.
    Converts custom Enum classes to their `.value`.
    
    Parameters
    ----------
    obj: dict[str, any]
        Initially a log dictionary containing metadata and a scaffolded 
        subdictionary structure containing layer parameters. On recursive 
        calls can be an object of any type.
    
    Returns
    -------
    dict[str, any]
        The same log dictionary passed in, with non-serializable types 
        converted to json serializable equivalents.
    """

    if isinstance(obj, Enum):
        return obj.name
    
    elif isinstance(obj, dict):
        # Recursively serialize all dict values
        return {k: make_json_serializable(v) for k,v in obj.items()}
    
    elif isinstance(obj, list):
        # Recursively serialize all list values
        return [make_json_serializable(v) for v in obj]
    
    elif isinstance(obj, tuple):
        # Recursively serialize all tuple values
        return tuple(make_json_serializable(v) for v in obj)
    
    else:
        return obj

def write_experiment_log(layers:list[Layer], working_directory_path:str = (os.path.normpath(os.path.join(os.getcwd(), "data")))) -> None:
    """ Given a list of layers and the working data folder path, 
    write out a JSON file log containing all layers, TimingConfigurations 
    and layer parameters used in the last `apply_layers()` call. 

    Parameters
    ----------

    layers : list[Layer]
        A list of the Layer objects to be logged. Internally
        passed by `apply_layers()` after each successful 
        processing run

    working_directory_path : str
        A path string to the working data folder within your
        current working directory. Defaults to 
        os.path.join(os.getcwd(), "data").
    
    Returns
    -------
    None

    Raises
    ------
    OSError
        If unable to locate working_director_path, or 
        working_directory_path is an invalid path string.
    
    ValidationError
        If any of the parameter data pass is not JSON
        serializable.
    """
    if os.getenv("PYTEST_RUNNING") == "1":
        return
    else:
        if not os.path.isdir(working_directory_path):
            raise OSError(message=f"Unable to locate the input {os.path.basename(working_directory_path)} directory. Please call make_output_paths() to initialise the working directory.")

        # Creating a unique file identifier
        timestamp = datetime.now().isoformat(timespec='seconds')
        output_path = os.path.join(working_directory_path, "logs")

        layer_dict = {}
        for layer in layers:
            layer_type = type(layer).__name__
            parameters = layer.get_layer_parameters()
            
            if parameters.get("landmark_paths"):
                parameters.update({"landmark_paths":get_landmark_names(parameters.get("landmark_paths"))})
            parameters.update({"rise_curve":parameters.get("rise_curve").__name__})
            parameters.update({"fall_curve":parameters.get("fall_curve").__name__})

            layer_dict[layer_type] = parameters

        log_data = {
            "schema_version":"1.0",
            "timestamp":timestamp,
            "layers": layer_dict
        }

        log_data = make_json_serializable(log_data)

        # Attempt to validate json against schema
        try:
            schema_path = files("pyfame.schema").joinpath("manipulation_log.v1.schema.json")
            schema = json.load(open(schema_path))
            jsonschema.validate(instance=log_data, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Experiment log failed to validate: {e.message}.")

        file_id = timestamp.replace(":","-")
        filename = os.path.join(output_path, f"{file_id}.json")

        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)

__all__ = ["write_experiment_log"]