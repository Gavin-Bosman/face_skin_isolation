from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeInt, PositiveFloat, PositiveInt
from typing import Optional, List, Tuple, Any
from pyfame.landmark.get_landmark_coordinates import get_face_landmarker, get_landmarker_coordinates
from pyfame.landmark.facial_landmarks import *
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.file_access import get_video_capture, get_video_writer
from pyfame.utils.exceptions import *
from pyfame.utils.constants import *
from pyfame.analyse._optical_flow_utils import draw_legend, draw_scaled_flow_arrows
from pyfame.file_access.file_access_directories import create_output_directory
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
from datetime import datetime
import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.util import *

class SparseFlowAnalysisParameters(BaseModel):
    landmark_idx_to_track:Optional[List[NonNegativeInt]]
    max_points:PositiveInt
    point_quality_threshold:PositiveFloat = 0.3
    min_point_distance:NonNegativeInt = 7
    pixel_neighborhood_size:Tuple[NonNegativeInt, NonNegativeInt] = (5,5)
    search_window_size:Tuple[NonNegativeInt, NonNegativeInt] = (15,15)
    max_pyramid_level:NonNegativeInt = 2
    max_iterations:PositiveInt = 10
    flow_accuracy_threshold:PositiveFloat
    arrow_thickness:PositiveInt
    legend:bool = True
    legend_position:str = "top-left"
    precise_colour_scale:bool = False
    stats_detail_level:str
    frame_step:PositiveInt = 1

    @field_validator("point_quality_threshold", "flow_accuracy_threshold")
    @classmethod
    def check_normal_range(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not (0.0 < value <= 1.0):
            raise ValueError(f"Parameter {field_name} must lie in the normalised range 0.0-1.0.")
        
        return value
    
    @field_validator("stats_detail_level")
    @classmethod
    def check_valid_stat_type(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if value not in {"summary", "full"}:
            raise ValueError(f"Unrecognized value for parameter {field_name}.")
        
        return value
    
    @field_validator("legend_position")
    @classmethod
    def check_legend_pos(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if value not in {"top-left", "bottom-left", "top-right", "bottom-right"}:
            raise ValueError(
                f"Unrecognized value provided for parameter {field_name}."
                f" {field_name} must be one of: 'top-left', 'bottom-left', 'top-right', 'bottom-right'."
            )
        
        return value
    
def precompute_colour_scale(file_path:str, ) -> tuple[float,float,Any]:
    """ Performs a shallow pass through the input file, using the
    minimum and maximum sparse optical flow magnitudes to compute 
    the data range for the visualization colour scale.

    This method is invoked internally by both `analyse_optical_flow_dense`
    and `analyse_optical_flow_sparse`.

    Parameters
    ----------

    file_path : str
        A path string to the video file to be analysed.
    
    Returns
    -------

    scale_values : tuple
        A tuple containing the min/max optical flow 
        magnitudes, and the normalization method.
    """

    print("------------------Scale precompute pass--------------------")
    results, _ = analyse_optical_flow_sparse(
        pd.DataFrame({"Absolute Path" : [file_path]}),
        max_points=10,
        stats_detail_level="full",
        frame_step=15
    )
    print("-----------------------------------------------------------")

    result_df = list(results.values())[0]
    mean_magnitude = result_df["magnitude"].mean()
    std_magnitude = result_df["magnitude"].std()

    magnitude_min = float(max(0.0, mean_magnitude - std_magnitude))
    magnitude_max = float(mean_magnitude + std_magnitude)
    norm = mpcolors.Normalize(vmin=magnitude_min, vmax=magnitude_max)

    return (magnitude_min, magnitude_max, norm)

def analyse_optical_flow_sparse(file_paths:pd.DataFrame, landmark_idx_to_track:list[int]|None = None, max_points:int = 20, 
                                flow_accuracy_threshold:float = 0.03, stats_detail_level:str = "summary", frame_step:int = 5, 
                                output_visualization:bool = False, display_legend:bool = True, legend_position:str = "top-left",
                                arrow_thickness:int = 2, precise_colour_scale:bool = False) -> tuple[dict[str, pd.DataFrame], str | None]:
    """Takes each input video file provided within input_directory, and 
    generates a sparse optical flow image, as well as a csv containing 
    periodically sampled flow vector data. This function makes use of 
    the Lucas-Kanadae optical flow algorithm, as well as the Shi-Tomasi 
    good-corners algorithm to identify and track relevant points in the 
    input video. Alternatively, specific facial landmarks to track can 
    be passed in via landmarks_to_track.
    
    Parameters
    ----------
    file_paths : DataFrame
        A 2-column dataframe consisting of absolute and relative file 
        paths.
    
    landmark_idx_to_track : list of int
        A list of mediapipe FaceMesh landmark id's, specifying relevant 
        facial landmarks to track.
    
    max_points : int
        The maximum number of corners or "good points" for the Shi-Tomasi 
        corners algorithm.

    flow_accuracy_threshold : float
        A termination criteria for Lucas-Kanadae optical flow; the algorithm 
        will continue to iterate until this threshold is reached.

    stats_detail_level : str
        Either "summary" specifying summary statisitics or "full" specifying 
        full descriptive output for each vector.
    
    frame_step : int
        The number of frames between successive optical flow calculations. 
        The flow values will be more consistent and robust as you increase 
        this parameter. 
    
    output_visualization : bool
        A boolean flag indicating whether to output a visualization video.
    
    display_legend : bool
        A boolean flag indicating whether to display a legend in the 
        visualization.

    legend_position : str
        A string indicating where in the visualization output the legend 
        should be placed. One of ["top-left", "top-right", "bottom-left",
        "bottom-right"].
    
    precise_colour_scale : bool
        A boolean flag indicating whether to do an initial precompute 
        pass using sparse optical flow to get a rough estimate of the 
        vector magnitude ranges for the visualizations colour scale.
    
    Returns
    -------

    dict[str, pandas.Dataframe]

    Raises
    ------

    ValueError
        Thrown by the pydantic model when invalid parameters are passed 
        to the method.
    
    FileReadError
        When the working directory path; or any of its required sub-paths 
        cannot be located. Additionally, if any tracking errors are 
        encountered mid analysis.

    """
    
    # Validate and assign input parameters
    try:
        input_parameters = SparseFlowAnalysisParameters(
            landmark_idx_to_track=landmark_idx_to_track, 
            max_points=max_points,
            flow_accuracy_threshold=flow_accuracy_threshold,
            arrow_thickness=arrow_thickness,
            legend=display_legend,
            legend_position=legend_position,
            precise_colour_scale=precise_colour_scale,
            stats_detail_level=stats_detail_level,
            frame_step=frame_step
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {analyse_optical_flow_sparse.__name__}: {e}")
    
    search_window_size = input_parameters.search_window_size
    max_pyramid_level = input_parameters.max_pyramid_level
    max_iterations = input_parameters.max_iterations
    point_quality_threshold = input_parameters.point_quality_threshold
    min_point_distance = input_parameters.min_point_distance
    pixel_neighborhood_size = input_parameters.pixel_neighborhood_size
    
    # Defining the mediapipe facemesh task
    face_landmarker = get_face_landmarker()
    
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

    if not os.path.isdir(test_path):
        raise FileReadError(message=f"Unable to locate the input {root_directory} directory. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "raw")):
        raise FileReadError(message=f"Unable to locate the 'raw' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "processed")):
        raise FileReadError(message=f"Unable to locate the 'processed' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")

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
        capture = None

        # Using the file extension to sniff video codec or image container for images
        if extension not in [".mp4", ".mov"]:
            print(f"Skipping unparseable file {os.path.basename(file)}.")
            continue
        
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
        num_points = []
        full_stats = []
        
        # Defining persistent loop params
        counter = 1
        init_points = None
        old_gray = None
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

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize  = search_window_size,
            maxLevel = max_pyramid_level,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iterations, flow_accuracy_threshold))

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
            
            # Get the landmark screen coordinates
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            landmarker_coordinates = get_landmarker_coordinates(frame_rgb, face_landmarker, None, True, False)
            # Create face oval image mask
            face_mask = mask_from_landmarks(frame, LANDMARK_FACE_OVAL, landmarker_coordinates)
            output_img = frame.copy()

            # First iteration finds good corners (shi-tomasi)
            if counter == 1:
                # Get initial tracking points
                old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # If landmarks were provided 
                if landmark_idx_to_track is not None:
                    init_points = np.array([
                            [lm[0], lm[1]] 
                            for i, lm in enumerate(landmarker_coordinates) 
                            if i in landmark_idx_to_track
                        ], 
                        dtype=np.float32
                    )
                    init_points = init_points.reshape(-1,1,2)
                else:
                    init_points = cv.goodFeaturesToTrack(old_gray, max_points, point_quality_threshold, min_point_distance, pixel_neighborhood_size, mask=face_mask)
                
                if output_visualization: writer.write(frame)
                counter += 1
                pb.update(1)
                continue
            
            # Second iteration onwards computes flow values 
            if counter > 1:
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)

                # Calculate sparse optical flow
                cur_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, init_points, None, **lk_params)

                # Select good points
                good_new_points = None
                good_old_points = None
                if cur_points is not None:
                    good_new_points = cur_points[st==1]
                    good_old_points = init_points[st==1]
                else:
                    raise FileReadError(message="analyse_optical_flow_sparse encountered an error: Cannot track points at the current frame.")

                # Update previous points for next iter
                init_points = good_new_points.reshape(-1, 1, 2)

                old_coords = []
                new_coords = []
                arrows = []
                # Computed only for the current time window
                magnitudes = []
                angles = []

                # Draw optical flow vectors and write out values
                for i, (new, old) in enumerate(zip(good_new_points, good_old_points)):
                    x0, y0 = old.ravel()
                    x1, y1 = new.ravel()
                    dx = x1 - x0
                    dy = y1 - y0
                    mag = np.sqrt(dx**2 + dy**2)

                    old_coords.append((x0, y0))
                    new_coords.append((x1, y1))
                    magnitudes.append(mag)
                    angles.append(np.arctan2(dy, dx))
                    arrows.append((int(x0), int(y0), int(dx), int(dy), float(mag)))

                global_mags.extend(magnitudes)

                # store summary statistics
                if counter % frame_step == 0:
                    if stats_detail_level == "summary":
                        mean_mag = np.mean(magnitudes)
                        std_mag = np.std(magnitudes)
                        mean_angle = np.mean(angles)
                        std_angle = np.std(angles)

                        # Dataframes are immutable, so we need to store as lists during execution
                        timestamps.append(timestamp/1000)
                        mean_magnitudes.append(mean_mag)
                        std_magnitudes.append(std_mag)
                        mean_angles.append(mean_angle)
                        std_angles.append(std_angle)
                        num_points.append(len(good_new_points))
                        
                    else:
                        for i, (old,new) in enumerate(zip(old_coords, new_coords)):
                            sample_stats = []
                            sample_stats.extend([timestamp//1000, old[0], old[1], new[0], new[1], magnitudes[i], angles[i]])
                            full_stats.append(sample_stats)

                if output_visualization:
                    if not precise_colour_scale:
                        # Dynamic recomputing of max magnitude (to adjust for local maxima)
                        if magnitude_max is None or magnitude_min is None or timestamp >= next_update_timestamp:
                            mean_mag = float(np.mean(global_mags))
                            std_mag = float(np.std(global_mags))
                            magnitude_min = mean_mag - std_mag
                            magnitude_max = mean_mag + std_mag
                            norm = mpcolors.Normalize(vmin=magnitude_min, vmax=magnitude_max)
                            next_update_timestamp += rolling_time_window
                    
                    # Scale the arrow length to the facial width
                    y_whites, x_whites = np.where(face_mask > 0)
                    x_min = x_whites.min()
                    x_max = x_whites.max()
                    arrow_length = int((x_max - x_min) * 0.1)

                    # Colour scale arrows by magnitude
                    for (x0, y0, dx, dy, mag) in arrows:
                        colour = cmap(norm(mag))[:3]   # RGB in [0,1]
                        colour_bgr = tuple(int(255*c) for c in colour[::-1])
                        draw_scaled_flow_arrows(output_img, (x0, y0), (dx, dy), colour_bgr, arrow_length, arrow_thickness)

                    if display_legend:
                        draw_legend(
                            frame=output_img, 
                            vmin=magnitude_min,
                            vmax=magnitude_max,
                            legend_position=legend_position
                        )
                
                old_gray = gray_frame.copy()
                if output_visualization: writer.write(output_img)
                counter += 1

            pb.update(1)

        # Resolve skipped frames in the progress bar
        if pb.__getattribute__("n") < capture.get(cv.CAP_PROP_FRAME_COUNT):
            pb.update(capture.get(cv.CAP_PROP_FRAME_COUNT) - pb.__getattribute__("n"))

        pb.close()

        capture.release()
        if output_visualization: writer.release()

        # Create and return dataframe
        if stats_detail_level == "summary":
            output_df = pd.DataFrame({
                "timestamp":timestamps,
                "mean magnitude":mean_magnitudes,
                "deviation magnitude":std_magnitudes,
                "mean angle":mean_angles,
                "deviation angle":std_angles,
                "number of points":num_points
            })
            
            outputs.update({f"{filename}":output_df})
        else:
            cols = ["timestamp", "old x", "old y", "new x", "new y", "magnitude", "angle"]
            output_df = pd.DataFrame(full_stats, columns=cols)
            outputs.update({f"{filename}":output_df})
    
    if output_visualization: return (outputs, folder_name)
    else: return (outputs, None)

__all__ = ["analyse_optical_flow_sparse"]