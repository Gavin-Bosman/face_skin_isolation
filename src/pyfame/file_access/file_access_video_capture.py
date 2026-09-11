import cv2 as cv
import os
from pyfame.utils.exceptions import *
from pyfame.file_access.checks import *

def get_video_capture(file_path:str) -> cv.VideoCapture:
    """ Convenience wrapper function returning a cv2.VideoCapture instance.

    Parameters
    ----------

    file_path : str
        A path string to the video file to be read in.
    
    Returns
    -------
    cv2.VideoCapture

    Raises
    ------
    TypeError
        When invalid input parameter typings are passed.
    OSError
        Given an invalid or incomplete file path.

    """
    # Perform parameter checks
    check_type(file_path, [str])
    check_valid_path(file_path)
    check_is_file(file_path)
    
    # Isolate the file extension to verify its compatibility with cv2.VideoCapture
    filename, extension = os.path.splitext(os.path.basename(file_path))
    check_valid_file_extension(extension=extension, allowed_extensions=[".mp4", ".mov"])

    # Instantiate the VideoCapture, then check for errors before returning the instance
    vc = cv.VideoCapture(file_path)

    if not vc.isOpened():
        raise FileReadError("Function has encountered an error attempting to instantiate cv2.VideoCapture()"
                           f" over file {file_path}.")
    else:
        return vc

__all__ = ["get_video_capture"]