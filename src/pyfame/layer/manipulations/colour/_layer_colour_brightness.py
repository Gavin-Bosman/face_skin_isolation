from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError
from typing import Union, List, Tuple
from pyfame.landmark.facial_landmarks import LANDMARK_FACE_OVAL
from pyfame.layer._layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.utils.constants import *
import cv2 as cv
import numpy as np

class BrightnessParameters(BaseModel):
    """
    Configuration model defining the control parameters for 
    manipulating the brightness of a frame or image.

    This class inherits from pydantic's `BaseModel` to provide validation and 
    default handling of brightness parameters.

    Attributes
    ----------
    landmark_paths : list of list of tuple of int or list of tuple of int
        A list of one or more closed landmark paths representing the 
        region in which the manipulation will be applied.
    magnitude : float
        The degree by which to increase or decrease the brightness.
        Used as an additive factor to increase the global intensity of a frame
        or image. Accepts values in the range [-25.0, 25.0]. Magnitudes outside
        of this range tend to cause artifacts in the resulting image.
    brightness_scale : str
        The method by which the brightness is scaled. `relative` indicates
        the scaled brightness range begins relative to the provided image or 
        video frame, where `absolute` indicates the brightness is scaled to
        the full 0-255 range.
    """

    landmark_paths:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]
    magnitude:float
    brightness_scale:str

    @field_validator("magnitude")
    @classmethod
    def check_value_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if not (-100.0 <= value <= 100.0):
            raise ValueError(f"{field_name} must lie in the range [-100,100].")
        
        return value

    @field_validator("brightness_scale")
    @classmethod
    def check_accepted_values(cls, value, info:ValidationInfo):
        field_name = info.field_name
        if str.lower(value) not in {"relative", "absolute"}:
            raise ValueError(f"{field_name} must be one of `relative` or `absolute`.")

        return str.lower(value)

class LayerColourBrightness(Layer):
    """
    Manipulation layer that adjusts image brightness within a specified
    landmark-defined region over time.

    This layer applies an additive brightness adjustment to a region of
    interest defined by one or more facial landmark paths. The strength of
    the manipulation may be modulated over time using the timing and
    weighting configuration inherited from `Layer`.

    Parameters
    ----------
    timing_configuration : TimingConfiguration
        Timing configuration controlling onset, offset, rise/fall durations,
        and temporal weighting behavior.
    brightness_parameters : BrightnessParameters
        Configuration model specifying the landmark region(s) and brightness
        magnitude of the manipulation.

    Attributes
    ----------
    time_config : TimingConfiguration
        Timing configuration used by the layer.
    bright_params : BrightnessParameters
        Brightness-specific configuration parameters.
    landmark_paths : list of list of tuple of int or list of tuple of int
        Landmark paths defining the region in which brightness manipulation
        is applied.
    magnitude : float
        The degree by which to increase or decrease the brightness.
        Used as an additive factor to increase the global intensity of a frame
        or image. Accepts values in the range [-25.0, 25.0]. Magnitudes outside
        of this range tend to cause artifacts in the resulting image.
    brightness_scale : str
        The method by which the brightness is scaled. `relative` indicates
        the scaled brightness range begins relative to the provided image or 
        video frame, where `absolute` indicates the brightness is scaled to
        the full 0-255 range.

    Notes
    -----
    - Positive magnitude values increase brightness, while negative values
      decrease brightness.
    - The effective magnitude at a given time is scaled by the temporal
      weight computed from the timing configuration.
    - This layer supports continuous temporal weighting.
    """

    def __init__(self, timing_configuration:TimingConfiguration, brightness_parameters:BrightnessParameters):
        """
        Initialize a brightness manipulation layer.

        Parameters
        ----------
        timing_configuration : TimingConfiguration
            Timing configuration controlling when the brightness manipulation
            is applied and how it transitions on and off.
        brightness_parameters : BrightnessParameters
            Parameters defining the target landmark region(s) and the
            brightness adjustment magnitude.

        Notes
        -----
        - The timing configuration is passed to the superclass `Layer`.
        - A snapshot of the initial layer state is taken after initialization
          to allow safe resetting between independent applications.
        """
        self.time_config = timing_configuration
        self.bright_params = brightness_parameters

        # Initialise the superclass
        super().__init__(self.time_config)

        # Define class parameters
        self.landmark_paths = self.bright_params.landmark_paths
        self.magnitude = self.bright_params.magnitude
        self.brightness_scale = self.bright_params.brightness_scale
        self.has_brightness_been_sampled = False
        self.relative_shift_amount = 0.0

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self):
        """
        Indicate whether the layer supports temporal weighting.

        Returns
        -------
        bool
            `True` if the layer supports continuous rise/fall weighting,
            `False` if it operates as a binary on/off manipulation.
        """
        return True
    
    def get_layer_parameters(self) -> dict:
        """
        Return the parameters defining this layer.

        This method should expose all configurable parameters required
        to reproduce the layer's behavior.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values.
        """
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.bright_params.model_dump())
        self._layer_parameters["onset_time_msec"] = self.onset_t
        self._layer_parameters["offset_time_msec"] = self.offset_t
        return dict(self._layer_parameters)

    def apply_layer(self, landmarker_coordinates:list[tuple[int, int]], frame:cv.typing.MatLike, dt:float):
        """
        Apply the layer's manipulation to a single frame.

        Parameters
        ----------
        landmarker_coordinates : list of tuple of int
            Facial landmark coordinates associated with the current frame.
        frame : MatLike
            Input image frame to which the manipulation is applied.
        dt : float
            Current time (in milliseconds).

        Returns
        -------
        MatLike
            The manipulated frame.
        
        Raises
        ------
        ValueError
            Given invalid or unrecognized parameter values.
        """
        if dt is None:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())
        
        # Occurs when the dt < onset_time, or > offset_time
        if weight == 0.0:
            return frame
        
        # Mask out the region of interest
        mask = mask_from_landmarks(frame, self.landmark_paths, landmarker_coordinates)

        # Otsu thresholding to seperate foreground and background
        grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
        thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

        # Adding a temporary image border to allow for correct floodfill behaviour
        bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
        floodfilled = bordered_thresholded.copy()
        cv.floodFill(floodfilled, None, (0,0), 255)

        # Removing temporary border and creating foreground mask
        floodfilled = floodfilled[10:-10, 10:-10]
        floodfilled = cv.bitwise_not(floodfilled)
        foreground = cv.bitwise_or(thresholded, floodfilled)

        # Reshape the mask for compatibility with cv2.convertScaleAbs()
        # mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)
        h,s,v = cv.split(frame_hsv)

        if not self.has_brightness_been_sampled:
            v_mu = np.mean(v, where=mask.astype(bool))
            relative_range = 255.0 - v_mu
            self.relative_shift_amount = (self.magnitude * relative_range) / 100.0
            self.has_brightness_been_sampled = True

        if self.brightness_scale == "relative":
            v = np.where(mask == 255, v + (weight * self.relative_shift_amount), v)
            v = np.clip(v, 0, 255)
        else:
            shift_amount = (self.magnitude * 255.0) / 100
            v = np.where(mask == 255, v + (weight * shift_amount), v)
            v = np.clip(v, 0, 255)

        final_img_hsv = cv.merge([h,s,v])
        img_brightened = cv.cvtColor(final_img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
        img_brightened[foreground == 0] = frame[foreground == 0]
        return img_brightened

def layer_colour_brightness(timing_configuration:TimingConfiguration | None = None, landmark_paths:list[list[tuple[int,...]]] | list[tuple[int,...]] = LANDMARK_FACE_OVAL, 
                            magnitude:float = 15.0, brightness_scale:str = "relative") -> LayerColourBrightness:
    """
    Factory function for the brightness manipulation layer. `LayerColourBrightness`
    leverages the HSV colour space to perform value (or brightness) shifts in the 
    specified area. The default relative brightness scale scales the magnitude of
    the brightness shift relative to the input brightness, whereas the absolute
    brightness scale scales the magnitude of the brightness shift to the full 
    [0, 255] value range.

    Parameters
    ----------
    timing_configuration : TimingConfiguration or None, optional
        A pydantic model containing timing configurations controlling onset, 
        offset, rise/fall durations, and weighting curves. If `None`, a 
        default `TimingConfiguration` is instantiated. The default 
        instantiation assumes a constant rise and fall transition, onset at 
        0.0 and offset at the video's duration.
    landmark_paths : list of list of tuple of int or list of tuple of int, default=LANDMARK_FACE_OVAL
        A list of one or more closed landmark paths representing the 
        region in which the manipulation will be applied.
    magnitude : float, default=20.0
        The degree by which to increase or decrease the brightness.
        Used as an additive factor to increase the global intensity of a frame
        or image. Accepts values in the range [-25.0, 25.0]. Magnitudes outside
        of this range tend to cause artifacts in the resulting image.
    brightness_scale : str
        The method by which the brightness is scaled. `relative` indicates
        the scaled brightness range begins relative to the provided image or 
        video frame, where `absolute` indicates the brightness is scaled to
        the full 0-255 range.
    
    Returns
    -------
    LayerColourBrightness
        An instance of the brightness manipulation layer.
    
    Raises
    ------
    ValueError
        When provided invalid or unrecognized parameter values.
    
    """
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = BrightnessParameters(
            landmark_paths=landmark_paths, 
            magnitude=magnitude,
            brightness_mode=brightness_scale
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerColourBrightness.__name__}: {e}")
        

    return LayerColourBrightness(time_config, params)

__all__ = ["BrightnessParameters", "layer_colour_brightness"]