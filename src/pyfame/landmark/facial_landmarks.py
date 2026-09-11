from pandas import DataFrame

def create_landmark_path(landmark_set:list[int]) -> list[tuple]:
    """Given a list of facial landmarks (int), returns a list of tuples, creating a closed path in the form 
    [(a,b), (b,c), (c,d), ...]. This function allows the user to create custom facial landmark sets.
    
    Parameters
    ----------

    landmark_set: list of int
        A python list containing facial landmark indicies.
    
    Returns
    -------
        
    closed_path: list of tuple
        A list of tuples containing overlapping points, forming a path.
    """
    
    # Connvert the input list to a two-column dataframe
    landmark_dataframe = DataFrame([(landmark_set[i], landmark_set[i+1]) for i in range(len(landmark_set) - 1)], columns=['p1', 'p2'])
    closed_path = []

    # Initialise the first two points
    p1 = landmark_dataframe.iloc[0]['p1']
    p2 = landmark_dataframe.iloc[0]['p2']

    for i in range(0, landmark_dataframe.shape[0]):
        obj = landmark_dataframe[landmark_dataframe['p1'] == p2]
        p1 = obj['p1'].values[0]
        p2 = obj['p2'].values[0]

        current_route = (p1, p2)
        closed_path.append(current_route)
    
    return closed_path


# Pertinent MediaPipe Face Mesh landmark index sets.

FACE_OVAL_IDX: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 401, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 177, 137,
    234, 127, 162, 21, 54, 103, 67, 109, 10
]
"""MediaPipe Face Mesh landmark indices defining the outer facial oval.
The first landmark is repeated at the end to form a closed boundary."""

RIGHT_EYE_REGION_IDX: list[int] = [
    71, 68, 104, 69, 107, 55, 189, 244, 233, 232, 231, 230, 229, 228,
    31, 35, 156, 71
]
"""MediaPipe Face Mesh landmark indices defining the right eye region,
including the area surrounding the eye and eyebrow."""

RIGHT_EYE_IDX: list[int] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160,
    161, 246, 33
]
"""MediaPipe Face Mesh landmark indices defining the contour of the right eye.
The first landmark is repeated at the end to form a closed boundary."""

RIGHT_IRIS_IDX: list[int] = [469, 470, 471, 472, 469]
"""MediaPipe Face Mesh landmark indices defining the contour of the right iris.
The first landmark is repeated at the end to form a closed boundary."""

RIGHT_EYEBROW_IDX: list[int] = [
    46, 53, 52, 65, 55, 107, 66, 105, 63, 70, 46
]
"""MediaPipe Face Mesh landmark indices defining the right eyebrow region.
The first landmark is repeated at the end to form a closed boundary."""

LEFT_EYE_REGION_IDX: list[int] = [
    301, 298, 333, 299, 336, 285, 413, 464, 453, 452, 451, 450, 449, 448,
    261, 265, 383, 301
]
"""MediaPipe Face Mesh landmark indices defining the left eye region,
including the area surrounding the eye and eyebrow."""

LEFT_EYE_IDX: list[int] = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387,
    388, 466, 263
]
"""MediaPipe Face Mesh landmark indices defining the contour of the left eye.
The first landmark is repeated at the end to form a closed boundary."""

LEFT_IRIS_IDX: list[int] = [474, 475, 476, 477, 474]
"""MediaPipe Face Mesh landmark indices defining the contour of the left iris.
The first landmark is repeated at the end to form a closed boundary."""

LEFT_EYEBROW_IDX: list[int] = [
    276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276
]
"""MediaPipe Face Mesh landmark indices defining the left eyebrow region.
The first landmark is repeated at the end to form a closed boundary."""

NOSE_IDX: list[int] = [
    168, 193, 122, 196, 236, 198, 209, 49, 48, 219, 235, 60, 242, 2, 290,
    455, 294, 331, 429, 420, 456, 419, 351, 417, 168
]
"""MediaPipe Face Mesh landmark indices defining the standard nose region.
The indices form a closed boundary around the nose."""

NOSE_WIDE_IDX: list[int] = [
    168, 193, 122, 196, 174, 217, 209, 49, 129, 64, 98, 167, 164, 393,
    327, 294, 278, 279, 429, 437, 399, 419, 351, 417, 168
]
"""MediaPipe Face Mesh landmark indices defining an expanded nose region.
This boundary encompasses a wider area around the nose than ``NOSE_IDX``."""

MOUTH_IDX: list[int] = [
    164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106,
    43, 57, 186, 92, 165, 167, 164
]
"""MediaPipe Face Mesh landmark indices defining the mouth region.
The boundary encompasses the lips and surrounding mouth area."""

LIPS_OUTER_IDX: list[int] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
    267, 37, 39, 40, 185, 61
]
"""MediaPipe Face Mesh landmark indices defining the outer lip contour.
The first landmark is repeated at the end to form a closed boundary."""

LIPS_INNER_IDX: list[int] = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311,
    312, 13, 82, 81, 80, 191, 78
]
"""MediaPipe Face Mesh landmark indices defining the inner lip contour.
The first landmark is repeated at the end to form a closed boundary."""

LEFT_CHEEK_IDX: list[int] = [
    427, 411, 376, 352, 345, 340, 261, 448, 449, 450, 451, 452, 453, 412,
    419, 399, 437, 429, 279, 423, 426, 427
]
"""MediaPipe Face Mesh landmark indices outlining the left cheek region.
These indices provide the boundary used when constructing the corresponding cheek region."""

RIGHT_CHEEK_IDX: list[int] = [
    207, 187, 147, 123, 116, 111, 31, 228, 229, 230, 231, 232, 233, 188,
    196, 174, 217, 209, 49, 203, 206, 207
]
"""MediaPipe Face Mesh landmark indices outlining the right cheek region.
These indices provide the boundary used when constructing the corresponding cheek region."""

CHIN_IDX: list[int] = [
    43, 106, 182, 83, 18, 313, 406, 335, 273, 422, 430, 394, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 169, 210, 202, 43
]
"""MediaPipe Face Mesh landmark indices defining the chin region.
The indices form a closed boundary encompassing the lower portion of the face."""

HEMI_FACE_TOP_IDX: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 137, 234, 127, 162,
    21, 54, 103, 67, 109, 10
]
"""MediaPipe Face Mesh landmark indices defining the upper half-face region.
The indices form a closed boundary spanning the upper portion of the facial oval."""

HEMI_FACE_BOTTOM_IDX: list[int] = [
    366, 401, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
    136, 172, 58, 177, 137, 366
]
"""MediaPipe Face Mesh landmark indices defining the lower half-face region.
The indices form a closed boundary spanning the lower portion of the facial oval."""

HEMI_FACE_LEFT_IDX: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 401, 288, 397, 365,
    379, 378, 400, 377, 152, 10
]
"""MediaPipe Face Mesh landmark indices defining the left half-face region.
The facial midline and outer facial oval form the region boundary."""

HEMI_FACE_RIGHT_IDX: list[int] = [
    152, 148, 176, 149, 150, 136, 172, 58, 177, 137, 234, 127, 162, 21,
    54, 103, 67, 109, 10, 152
]
"""MediaPipe Face Mesh landmark indices defining the right half-face region.
The facial midline and outer facial oval form the region boundary."""


# Preconstructed face-region paths for use with facial manipulation functions.
# The following landmarks represent convex polygons.

LANDMARK_LEFT_EYE_REGION: list[tuple[int, int]] = create_landmark_path(
    LEFT_EYE_REGION_IDX
)
"""Closed landmark path defining the left eye region, including the surrounding
eye and eyebrow area. Constructed from ``LEFT_EYE_REGION_IDX``."""

LANDMARK_LEFT_EYE: list[tuple[int, int]] = create_landmark_path(LEFT_EYE_IDX)
"""Closed landmark path defining the left eye contour.
Constructed from ``LEFT_EYE_IDX``."""

LANDMARK_LEFT_IRIS: list[tuple[int, int]] = create_landmark_path(LEFT_IRIS_IDX)
"""Closed landmark path defining the left iris.
Constructed from ``LEFT_IRIS_IDX``."""

LANDMARK_LEFT_EYEBROW: list[tuple[int, int]] = create_landmark_path(
    LEFT_EYEBROW_IDX
)
"""Closed landmark path defining the left eyebrow region.
Constructed from ``LEFT_EYEBROW_IDX``."""

LANDMARK_RIGHT_EYE_REGION: list[tuple[int, int]] = create_landmark_path(
    RIGHT_EYE_REGION_IDX
)
"""Closed landmark path defining the right eye region, including the surrounding
eye and eyebrow area. Constructed from ``RIGHT_EYE_REGION_IDX``."""

LANDMARK_RIGHT_EYE: list[tuple[int, int]] = create_landmark_path(RIGHT_EYE_IDX)
"""Closed landmark path defining the right eye contour.
Constructed from ``RIGHT_EYE_IDX``."""

LANDMARK_RIGHT_IRIS: list[tuple[int, int]] = create_landmark_path(RIGHT_IRIS_IDX)
"""Closed landmark path defining the right iris.
Constructed from ``RIGHT_IRIS_IDX``."""

LANDMARK_RIGHT_EYEBROW: list[tuple[int, int]] = create_landmark_path(
    RIGHT_EYEBROW_IDX
)
"""Closed landmark path defining the right eyebrow region.
Constructed from ``RIGHT_EYEBROW_IDX``."""

LANDMARK_NOSE: list[tuple[int, int]] = create_landmark_path(NOSE_IDX)
"""Closed landmark path defining the standard nose region.
Constructed from ``NOSE_IDX``."""

LANDMARK_MOUTH: list[tuple[int, int]] = create_landmark_path(MOUTH_IDX)
"""Closed landmark path defining the mouth and its surrounding region.
Constructed from ``MOUTH_IDX``."""

LANDMARK_LIPS_OUTER_CONTOUR: list[tuple[int, int]] = create_landmark_path(
    LIPS_OUTER_IDX
)
"""Closed landmark path following the outer contour of the lips.
Constructed from ``LIPS_OUTER_IDX``."""

LANDMARK_LIPS_INNER_CONTOUR: list[tuple[int, int]] = create_landmark_path(
    LIPS_INNER_IDX
)
"""Closed landmark path following the inner contour of the lips.
Constructed from ``LIPS_INNER_IDX``."""

LANDMARK_FACE_OVAL: list[tuple[int, int]] = create_landmark_path(FACE_OVAL_IDX)
"""Closed landmark path defining the outer facial oval.
Constructed from ``FACE_OVAL_IDX``."""

LANDMARK_HEMI_FACE_TOP: list[tuple[int, int]] = create_landmark_path(
    HEMI_FACE_TOP_IDX
)
"""Closed landmark path defining the upper half of the face.
Constructed from ``HEMI_FACE_TOP_IDX``."""

LANDMARK_HEMI_FACE_BOTTOM: list[tuple[int, int]] = create_landmark_path(
    HEMI_FACE_BOTTOM_IDX
)
"""Closed landmark path defining the lower half of the face.
Constructed from ``HEMI_FACE_BOTTOM_IDX``."""

LANDMARK_HEMI_FACE_LEFT: list[tuple[int, int]] = create_landmark_path(
    HEMI_FACE_LEFT_IDX
)
"""Closed landmark path defining the left half of the face.
Constructed from ``HEMI_FACE_LEFT_IDX``."""

LANDMARK_HEMI_FACE_RIGHT: list[tuple[int, int]] = create_landmark_path(
    HEMI_FACE_RIGHT_IDX
)
"""Closed landmark path defining the right half of the face.
Constructed from ``HEMI_FACE_RIGHT_IDX``."""


# The following landmark regions are partially computed in place. Placeholder
# paths allow them to be passed to the facial-manipulation family of functions.
# These regions represent concave polygons.

LANDMARK_BOTH_CHEEKS: list[tuple[int, ...]] = [(0,)]
"""Placeholder landmark path representing both cheek regions.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_LEFT_CHEEK: list[tuple[int, ...]] = [(1,)]
"""Placeholder landmark path representing the left cheek region.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_RIGHT_CHEEK: list[tuple[int, ...]] = [(2,)]
"""Placeholder landmark path representing the right cheek region.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_CHEEKS_AND_NOSE: list[tuple[int, ...]] = [(3,)]
"""Placeholder landmark path representing the combined cheeks and nose region.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_BOTH_EYE_REGIONS: list[tuple[int, ...]] = [(4,)]
"""Placeholder landmark path representing both eye regions.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_FACE_SKIN: list[tuple[int, ...]] = [(5,)]
"""Placeholder landmark path representing the facial skin region.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_CHIN: list[tuple[int, ...]] = [(6,)]
"""Placeholder landmark path representing the chin region.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_BOTH_EYES: list[tuple[int, ...]] = [(8,)]
"""Placeholder landmark path representing both eyes.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_BOTH_IRISES: list[tuple[int, ...]] = [(9,)]
"""Placeholder landmark path representing both irises.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_BOTH_EYEBROWS: list[tuple[int, ...]] = [(10,)]
"""Placeholder landmark path representing both eyebrows.
The complete concave region is computed when it is used by a facial manipulation."""

LANDMARK_LIPS: list[tuple[int, ...]] = [(11,)]
"""Placeholder landmark path representing the complete lip region.
The complete concave region is computed when it is used by a facial manipulation."""

CONCAVE_LANDMARKS: list[list[tuple[int, ...]]] = [
    LANDMARK_BOTH_CHEEKS,
    LANDMARK_LEFT_CHEEK,
    LANDMARK_RIGHT_CHEEK,
    LANDMARK_CHEEKS_AND_NOSE,
    LANDMARK_BOTH_EYE_REGIONS,
    LANDMARK_FACE_SKIN,
    LANDMARK_CHIN,
    LANDMARK_BOTH_EYES,
    LANDMARK_BOTH_IRISES,
    LANDMARK_BOTH_EYEBROWS,
    LANDMARK_LIPS,
]
"""Collection of landmark placeholders corresponding to facial regions that
require concave region construction at runtime."""