from pydantic import BaseModel, ValidationError, field_validator, ValidationInfo, PositiveInt, NonNegativeInt, PositiveFloat, NonNegativeFloat
from pyfame.landmark.facial_landmarks import *
from pyfame.file_access import get_video_capture, get_video_writer
from pyfame.utils.exceptions import *
from pyfame.utils.constants import *
from pyfame.file_access.file_access_directories import create_output_directory
from pyfame.analyse._optical_flow_utils import draw_legend
from pyfame.analyse.analyse_optical_flow_sparse import precompute_colour_scale
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
from datetime import datetime
import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.util import *

class DenseFlowAnalysisParameters(BaseModel):
    pixel_neighborhood_size:PositiveInt = 5
    search_window_size:PositiveInt = 15
    max_pyramid_level:NonNegativeInt = 2
    pyramid_scale:PositiveFloat = 0.5
    max_iterations:PositiveInt = 10
    gaussian_deviation:NonNegativeFloat = 1.2
    frame_step:PositiveInt = 5
    precise_colour_scale:bool = True
    display_legend:bool
    legend_position:str="top-left"

    @field_validator("legend_position")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if value not in {"top-left", "bottom-left", "top-right", "bottom-right"}:
            raise ValueError(
                f"Unrecognized value provided for parameter {field_name}."
                f" {field_name} must be one of: 'top-left', 'bottom-left', 'top-right', 'bottom-right'."
            )
        
        return value
    
def analyse_optical_flow_dense(file_paths:pd.DataFrame, frame_step:int = 5, output_visualization:bool = False, precise_colour_scale:bool = False,
                               display_legend:bool = True, legend_position:str = "top-left") -> tuple[dict[str, pd.DataFrame], str | None]:
    '''Takes an input video file, and computes the dense optical flow, outputting the 
    aggregate vector magnitudes to a csv file. Optionally, output a visualization of 
    the dense optical flow into the same folder as the csv files, under /visualization.
    Dense optical flow uses Farneback's algorithm to track every point within a frame.

    Parameters
    ----------

    file_paths : pandas.DataFrame
        An Nx2 dataframe of absolute and relative file paths, returned by 
        the make_paths() function.
    
    frame_step : int
        The number of frames between successive optical flow calculations. 
        The flow values will be more consistent and robust as you increase 
        this parameter, but computation time will increase proportionally.

    output_visualization : bool
        A boolean flag indicating whether to output a visualization video.
    
    precise_colour_scale : bool
        A boolean flag indicating whether to do an initial precompute pass 
        using sparse optical flow to get a rough estimate of the vector 
        magnitude ranges for the visualizations colour scale. 
    
    display_legend : bool
        A boolean flag indicating whether to display a legend in the 
        visualization.

    legend_position : str
        A string indicating where in the visualization output the legend 
        should be placed. One of ["top-left", "top-right", "bottom-left",
        "bottom-right"].
    
    Returns
    -------
    dict[str, pandas.DataFrame]

    Raises
    ------
    ValidationError
        Thrown by the pydantic model when invalid parameters are passed to the method.
    
    FileReadError
        When the working directory path; or any of its required sub-paths cannot be located. 

    UnrecognizedExtensionError
        If an image file is passed; Farneback's dense flow requires video files.
    '''

    # Validate and assign input parameters
    try:
        input_parameters = DenseFlowAnalysisParameters(
            frame_step=frame_step,
            precise_colour_scale=precise_colour_scale,
            display_legend=display_legend,
            legend_position=legend_position
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {analyse_optical_flow_dense.__name__}: {e}")

    pyramid_scale = input_parameters.pyramid_scale
    max_pyramid_level = input_parameters.max_pyramid_level
    search_window_size = input_parameters.search_window_size
    max_iterations = input_parameters.max_iterations
    pixel_neighborhood_size = input_parameters.pixel_neighborhood_size
    gaussian_deviation = input_parameters.gaussian_deviation

    # Extracting the i/o paths from the file_paths dataframe
    absolute_paths = file_paths["Absolute Path"]

    norm_path = os.path.normpath(absolute_paths.iloc[0])
    norm_cwd = os.path.normpath(os.getcwd())
    rel_dir_path, *_ = os.path.split(os.path.relpath(norm_path, norm_cwd))
    parts = rel_dir_path.split(os.sep)
    root_directory = None

    if parts is not None:
        root_directory = parts[0]
    
    if root_directory is None:
        root_directory = "data"
    
    test_path = os.path.join(norm_cwd, root_directory)

    # Test the path and its extensions exist in the file system
    if not os.path.isdir(test_path):
        raise FileReadError(message=f"Unable to locate the input {root_directory} directory. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "raw")):
        raise FileReadError(message=f"Unable to locate the 'raw' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "analysis")):
        raise FileReadError(message=f"Unable to locate the 'analysis' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    
    output_root = os.path.join(test_path, "analysis")
    timestamp = datetime.now().isoformat(timespec='seconds')
    folder_name = timestamp.replace(":","-")

    # Create the outputs dict outside of the main loop so it maintains a larger scope
    outputs = {}

    for file in tqdm(
            absolute_paths,
            total=len(absolute_paths),
            desc="Files processed",
            bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}',
            position=0,
            dynamic_ncols=True
        ):
        
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))

        # Using the file extension to sniff video codec or image container for images
        if extension not in [".mp4", ".mov"]:
            raise ValueError(f"Cannot compute optical flow on non-video filetype: '{extension}'.")
        
        # Instantiating video read/writers
        capture = get_video_capture(file)

        if output_visualization:
            visualization_output_path = create_output_directory(create_output_directory(output_root, folder_name), "visualization")
            size = (int(capture.get(3)), int(capture.get(4)))
            writer = get_video_writer(
                file_path=os.path.join(visualization_output_path, f"{filename}{extension}"),
                frame_size=size
            )

        # creating lists to store output data
        timestamps = []
        mean_magnitudes = []
        std_magnitudes = []
        mean_angles = []
        std_angles = []

        # Defining persistent loop params
        counter = 1
        previous_grey_frame = None
        cmap = cm.get_cmap("viridis")
        global_mags = []
        magnitude_min = None
        magnitude_max = None
        norm = None
        # Used to recompute the max magnitude every 500 msec or ~15 frames
        rolling_time_window = 500.0
        next_update_timestamp = rolling_time_window

        if precise_colour_scale:
            magnitude_min, magnitude_max, norm = precompute_colour_scale(file)
        
        pb = tqdm(
            total=capture.get(cv.CAP_PROP_FRAME_COUNT), 
            desc="Frames analysed",
            bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', 
            colour="blue",
            position=1,
            dynamic_ncols=True
        )

        # Main Processing loop
        while True:

            success, frame = capture.read()
            if not success:
                break  
            dt = capture.get(cv.CAP_PROP_POS_MSEC)

            if counter == 1:
                previous_grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                
                if output_visualization:
                    viridis = cmap(np.zeros(frame.shape[:2], dtype=float))
                    viridis_rgb = viridis[:, :, :3]
                    viridis_bgr = viridis_rgb[:, :, ::-1]
                    output_img = (viridis_bgr * 255).astype(np.uint8)
                    writer.write(output_img)

                counter += 1
                pb.update(1)
                continue

            if counter > 1:
                # Used to recompute the max magnitude every 500 msec or ~15 frames
                timestamp_msec = dt
                rolling_time_window = 500

                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                if output_visualization or counter % frame_step == 0:
                    # Calculate dense optical flow
                    flow = cv.calcOpticalFlowFarneback(
                        previous_grey_frame, 
                        grey_frame,
                        None, 
                        pyramid_scale, 
                        max_pyramid_level, 
                        search_window_size, 
                        max_iterations, 
                        pixel_neighborhood_size, 
                        gaussian_deviation, 
                        0
                    )

                    # Get vector magnitudes and angles
                    magnitudes, angles = cv.cartToPolar(flow[...,0],flow[...,1])

                # Update the prev grey frame after calculating the current frame's flow
                previous_grey_frame = grey_frame.copy()

                if output_visualization:
                    # Collect global mags
                    global_mags.extend(magnitudes.flatten())
                    
                    # Updating the magnitude range of the colour scale
                    if not precise_colour_scale:
                        if magnitude_max is None or magnitude_min is None or timestamp_msec >= next_update_timestamp:
                            mean_mag = float(np.mean(global_mags))
                            std_mag = float(np.std(global_mags))
                            magnitude_min = mean_mag - std_mag
                            magnitude_max = mean_mag + std_mag
                            norm = mpcolors.Normalize(vmin=magnitude_min, vmax=magnitude_max)
                            next_update_timestamp += rolling_time_window

                    # Normalise magnitudes to [0,1]
                    normal_mags = norm(magnitudes)

                    # Map magnitudes to viridis colour scale
                    viridis = cmap(normal_mags)
                    viridis_rgb = viridis[:, :, :3]
                    viridis_bgr = viridis_rgb[:, :, ::-1]
                    output_img = (viridis_bgr * 255).astype(np.uint8)

                    if display_legend:
                        draw_legend(
                            frame=output_img, 
                            vmin=magnitude_min, 
                            vmax=magnitude_max, 
                            legend_position=legend_position
                        )

                    writer.write(output_img)

                if counter % frame_step == 0:
                    # Get magnitude/angle means and distribution
                    mean_mag = np.mean(magnitudes)
                    std_mag = np.std(magnitudes)
                    mean_angle = np.mean(angles)
                    std_angle = np.std(angles)

                    # Dataframes are immutable, so we need to store as lists during execution
                    timestamps.append(timestamp_msec/1000)
                    mean_magnitudes.append(mean_mag)
                    std_magnitudes.append(std_mag)
                    mean_angles.append(mean_angle)
                    std_angles.append(std_angle)

                pb.update(1)
                counter += 1
        
        # Resolve skipped frames in the progress bar
        if pb.__getattribute__("n") < capture.get(cv.CAP_PROP_FRAME_COUNT):
            pb.update(capture.get(cv.CAP_PROP_FRAME_COUNT) - pb.__getattribute__("n"))
            
        pb.close()

        capture.release()
        if output_visualization: writer.release()

        output_df = pd.DataFrame({
            "timestamp":timestamps,
            "mean magnitude":mean_magnitudes,
            "deviation magnitude":std_magnitudes,
            "mean angle":mean_angles,
            "deviation angle":std_angles
        })
        outputs.update({f"{filename}":output_df})
    
    if output_visualization:
        return (outputs, folder_name)
    else:
        return (outputs, None)
    
__all__ = ["analyse_optical_flow_dense"]