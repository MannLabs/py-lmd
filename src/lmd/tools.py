import os
import warnings

import numpy as np

from lmd._utils import _download_glyphs
from lmd.lib import Collection, Shape


def _get_rotation_matrix(angle: float):
    """Returns a rotation matrix for clockwise rotation.

    Args:
        angle (float): Rotation in radian.
    Returns:
        np.ndarray: Matrix in the shape of (2, 2).
    """

    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def glyph_path(glyph):
    """Returns the path for a glyph of interest. Raises a NotImplementedError if an unknown glyph is requested.

    Args:
        glyph (str): Single glyph as string.

    Returns:
        str: Path for the glyph.
    """
    file_path = _download_glyphs()

    svg_path = file_path / f"{glyph}.svg"

    # Check if glyph exists, raise NotImplementedError if not
    if os.path.isfile(svg_path):
        return svg_path
    else:
        raise NotImplementedError(f"You tried to load the glyph {glyph}. This has not been implemented yet.")


# TODO: Default divisor=3 is unintuitive and should be revisited. Current default is applied to prevent breaking changes (Shape.svg_to_lmd default is 3)
def glyph(
    glyph: str,
    offset: np.ndarray = np.array([0, 0]),
    rotation: float = 0,
    divisor: int = 3,
    multiplier: float = 1,
    **kwargs,
) -> Collection:
    """Get an uncalibrated lmd.lib.Collection for a glyph of interest.

    Args:
        glyph: Single glyph as string.
        offset: Location of the glyph based on the top left corner. Defaults to no offset.
        rotation: Rotation of glyph in radians
        divisor: Parameter which determines the resolution when creating a polygon from a SVG. A larger divisor will lead to fewer datapoints for the glyph.
        multiplier: Scaling parameter for defining the size of the glyph. The default height of a glyph is 10 units.

    Returns:
        lmd.lib.Collection: Uncalibrated Collection which contains the Shapes for the glyph.
    """
    svg_path = glyph_path(glyph)

    # TODO: Document local_multiplier
    local_multiplier = 0.2
    multiplier = multiplier * local_multiplier

    shapefile = Collection()
    shapefile.svg_to_lmd(
        svg_path,
        offset=offset,
        rotation_matrix=_get_rotation_matrix(rotation),
        divisor=divisor,
        multiplier=multiplier,
        **kwargs,
    )

    return shapefile


# TODO: Default divisor=3 is unintuitive and should be revisited. Current default is applied to prevent breaking changes (Shape.svg_to_lmd default is 3)
def text(
    text: str,
    offset: np.ndarray = np.array([0, 0]),
    divisor: int = 3,
    multiplier: float = 1,
    rotation: float = 0,
    **kwargs,
) -> Collection:
    """Get an uncalibrated lmd.lib.Collection for a text.

    Args:
        text: Text as string.
        offset: Location of the text based on the top left corner. Defaults to no offset.
        divisor: Parameter which determines the resolution when creating a polygon from a SVG. A larger divisor will lead to fewer datapoints for the glyph.
        multiplier: Scaling parameter for defining the size of the text. The default height of a glyph is 10 units.
        rotation: Rotation of glyph in radians

    Returns:
        Uncalibrated Collection which contains the Shapes for the text.

    Example:

        .. code-block:: python

            import numpy as np
            from lmd.lib import Collection, Shape
            from lmd import tools

            calibration = np.array([[0, 0], [0, 100], [100, 50]])
            my_first_collection = Collection(calibration_points=calibration)

            identifier_1 = tools.text("0456_B2", offset=np.array([30, 40]), rotation=-np.pi / 4)
            my_first_collection.join(identifier_1)
            my_first_collection.plot(calibration=True)

        .. image:: images/fig10.png
           :scale: 100%
    """

    # Convert text to str and assert
    text = str(text)

    # delta offset for every digit
    delta = np.array([10, 0]) @ _get_rotation_matrix(rotation) * multiplier

    heap = Collection()

    # enumerate all glyphs and append shapes
    for i, current_glyph in enumerate(text):
        current = glyph(
            current_glyph,
            offset=offset + i * delta,
            multiplier=multiplier,
            divisor=divisor,
            rotation=rotation,
            **kwargs,
        )
        heap.join(current)

    return heap


def rectangle(
    width: float,
    height: float,
    offset: np.ndarray = np.array([0, 0]),
    rotation: float = 0,
    rotation_offset: np.ndarray = np.array([0, 0]),
) -> Shape:
    """Get a lmd.lib.Shape for rectangle of choosen dimensions.

    Args:
        width: Width of the rectangle.
        height: Height of the rectangle.
        offset: Location of the rectangle based on the center. Defaults to no offset
        rotation: Rotation in radian.
        rotation_offset: Location of the center of rotation relative to the center of the rectangle. Defaults to no offset.

    Returns:
        Shape which contains the rectangle.

    Example:
    """
    offset = np.array(offset)
    rotation_offset = np.array(rotation_offset)
    rotation_mat = _get_rotation_matrix(rotation)

    points = np.array(
        [
            [-width / 2, -height / 2],
            [-width / 2, height / 2],
            [width / 2, height / 2],
            [width / 2, -height / 2],
            [-width / 2, -height / 2],
        ]
    )

    points = (points + rotation_offset) @ rotation_mat - rotation_offset + offset
    return Shape(points)


def ellipse(
    major_axis: float,
    minor_axis: float,
    offset: np.ndarray = np.array([0, 0]),
    rotation: float = 0,
    polygon_resolution: float = 1,
) -> Shape:
    """Get a lmd.lib.Shape for ellipse of choosen dimensions.

    Args:
        major_axis: Major axis of the ellipse. The major axis is defined from the center to the perimeter and therefore half the diameter. The major axis is placed along the x-axis before rotation.
        minor_axis: Minor axis of the ellipse. The minor axis is defined from the center to the perimeter and therefore half the diameter. The minor axis is placed along the y-axis before rotation.
        offset: Location of the ellipse based on the center given in the form of `(x, y)`. Defaults to no offset.
        rotation: Clockwise rotation in radian.
        polygon_resolution: The polygon resolution defines how far the vertices should be spaced on average. A polygon_resolution of 10 will place a vertex on average every ten units.

    Returns:
        Shape which contains the ellipse.

    Example:

        .. code-block:: python

            import numpy as np
            from lmd.lib import Collection, Shape
            from lmd import tools

            calibration = np.array([[0, 0], [0, 100], [50, 50]])
            my_first_collection = Collection(calibration_points=calibration)

            my_ellipse = tools.ellipse(20, 10, offset=(30, 50), polygon_resolution=5, rotation=1.8 * np.pi)
            my_first_collection.add_shape(my_ellipse)
            my_first_collection.plot(calibration=True)

        .. image:: images/tools.ellipse.example.png
           :scale: 100%
    """

    if polygon_resolution == 0:
        raise ValueError("Polygon resolution has to be larger than 0")

    a = minor_axis
    b = major_axis

    h = ((a - b) ** 2) / ((a + b) ** 2)

    # Approximation of circumference according to Ramanujan
    P = np.pi * (a + b) * (1 + (3 * h / (10 + np.sqrt(4 - 3 * h))))

    # The number of vertices is given by the circumference and the polygon_resolution
    n_vertices = np.round(P / polygon_resolution).astype(int)

    # The radian space returns a list on n_vertices spaced between 0 and 2*pi
    # datapoints are therefore spaced according to equal angle, not equal distance on the perimeter!
    # If anybody has suggestions how to improve this, feel free to contribute (I should check Kepler)
    radian_space = np.linspace(0, 2 * np.pi, n_vertices)

    unit_circle = np.array([np.cos(radian_space), np.sin(radian_space)]).T

    # Scaling the unit circle gives the ellipse
    ellipse_scale = np.array([major_axis, minor_axis])
    ellipse = unit_circle * ellipse_scale @ _get_rotation_matrix(rotation) + offset

    return Shape(ellipse)


def make_cross(center: np.ndarray, arms: np.ndarray, width: float, dist: float) -> Collection:
    """Generate lmd.lib.Shapes to represent a crosshair and add them to an exisiting lmd.lib.Collection.

    Args:
        center: center of the new crosshair
        arms: length of the individual arms [top, right, bottom, left]
        width: width of each individual element of the crosshair
        dist: Controls the gap between the center dot and the arms. The inner edge of each arm sits at a distance of ``2 * dist`` from the center, so larger values spread the arms further out.

    Returns:
        Uncalibrated Collection which contains the Shapes for the calibration cross.

    Example:

        .. code-block:: python

            import numpy as np
            from lmd.lib import Collection, Shape
            from lmd import tools

            calibration = np.array([[0, 0], [0, 100], [50, 50]])
            my_first_collection = Collection(calibration_points=calibration)

            cross_1 = tools.make_cross([20, 20], [50, 50, 50, 50], 1, 10)
            my_first_collection.join(cross_1)
            my_first_collection.plot(calibration=True)

        .. image:: images/tools.makeCross.example.png
           :scale: 50%

    """
    heap = Collection()

    # generate central dot and add to collection
    centerdot = rectangle(width, width, offset=center)
    heap.add_shape(centerdot)

    # generate top arm
    local_transform = np.array([0, +arms[0] / 2 + 2 * dist])
    offset_toparm = center + local_transform
    toparm = rectangle(width, arms[0], offset=offset_toparm)
    heap.add_shape(toparm)

    # generate right arm
    local_transform = np.array([+arms[1] / 2 + 2 * dist, 0])
    offset_rightarm = center + local_transform
    toparm = rectangle(arms[1], width, offset=offset_rightarm)
    heap.add_shape(toparm)

    # generate bottom arm
    local_transform = np.array([0, -arms[2] / 2 - 2 * dist])
    offset_bottomarm = center + local_transform
    bottomarm = rectangle(width, arms[2], offset=offset_bottomarm)
    heap.add_shape(bottomarm)

    # generate left arm
    local_transform = np.array([-arms[3] / 2 - 2 * dist, 0])
    offset_leftarm = center + local_transform
    leftarm = rectangle(arms[3], width, offset=offset_leftarm)
    heap.add_shape(leftarm)

    return heap


# TODO: Remove with version 2.0.0
def makeCross(center: np.ndarray, arms: np.ndarray, width: float, dist: float) -> Collection:
    """Generate lmd.lib.Shapes to represent a crosshair and add them to an exisiting lmd.lib.Collection.

    Notes
    -----
    Deprecated and will be removed in version 2.0.0. Use :func:`lmd.tools.make_cross` instead.

    See Also
    --------
    :func:`lmd.tools.make_cross`
    """
    warnings.warn(
        "lmd.tools.makeCross has been renamed to lmd.tools.make_cross. "
        "The function is deprecated and will be removed in v2.0.0. "
        "Please use: `from lmd.tools import make_cross` for equivalent functionality",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_cross(center=center, arms=arms, width=width, dist=dist)
