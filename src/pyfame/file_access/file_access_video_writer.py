import cv2 as cv
from pyfame.utils.exceptions import *
from pyfame.file_access.checks import *

def get_video_writer(file_path:str, frame_size:tuple[int,int], video_codec:str = 'mp4v', frame_rate:float = 30, isColor:bool = True) -> cv.VideoWriter:
    """ Convenience wrapper function returning a cv2.VideoWriter object.

    Parameters
    ----------
    file_path : str
        The file path to be opened for writing.
    
    frame_size : tuple[int,int]
        A tuple of integers containing frame dimensions of the 
        frames to be written.
    
    video_codec : str
        A cv2 compatible video codec string.
    
    frame_rate : float
        The frame rate of the output video file.
    
    isColor : bool
        A boolean indicating if the frames being written 
        are in black/white or colour.
    
    Returns
    -------
    cv2.VideoWriter

    Raises
    ------
    TypeError
        On invalid parameter typings.
    ValueError
        when unexpected or incompatible values are passed.

    """
    # Perform parameter checks
    check_type(file_path, [str])
    check_valid_path(file_path)

    check_type(frame_size, [tuple])
    check_type(frame_size, [int], iterable=True)

    check_type(video_codec, [str])
    check_value(video_codec, ['mp4v', 'XVID'])

    check_type(frame_rate, [float])
    check_value(frame_rate, min=1, max=60)

    check_type(isColor, [bool])

    # Create VideoWriter instance
    vw = cv.VideoWriter(file_path, cv.VideoWriter.fourcc(*video_codec), frame_rate, frame_size, isColor=isColor)
    # Check for any errors with object creation before returning the videoWriter instance
    if not vw.isOpened():
        raise FileWriteError("Function encountered an error attempting to instantiate "
                            f"cv.VideoWriter() over file {file_path}.")
    else:
        return vw
    
__all__ = ["get_video_writer"]