"""Core geometrical objects"""

from __future__ import annotations

import re
import warnings
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from lxml import etree as ET
from svgelements import SVG


class Shape:
    """Class for creating a single shape object."""

    def __init__(
        self,
        points: np.ndarray = np.empty((1, 2)),
        well: str | None = None,
        name: str | None = None,
        orientation_transform=None,
        **custom_attributes: dict[str, str],
    ):
        """Class for creating a single shape.

        Args:
            points: Array or list of lists in the shape of `(N,2)`. Contains the points of the polygon forming a shape.

            well: Well in which to sort the shape after cutting. For example A1, A2 or B3.

            name: Name of the shape.

            custom_attributes: Custom shape metadata that will be added as additional xml-element to the shape
                Values be implicitly converted to strings.
        """
        # Orientation transform of shapes
        self.orientation_transform: np.ndarray | None = orientation_transform

        # Although a numpy array is recommended, list of lists is accepted
        points = np.array(points)

        # Assert correct dimensions
        point_shapes = points.shape
        if (points.ndim != 2) or (point_shapes[1] != 2):
            raise ValueError(
                f"Shape {name}: Shape dimensionality is not valid. Please provide a numpy array of shape (N, 2)"
            )

        if len(points) < 3:
            raise ValueError(
                f"Shape {name}: Valid shape must contain at least 3 points, but only contains {len(points)}"
            )

        self.points: np.ndarray = points

        self.name: str | None = name
        self.well: str | None = well

        self.custom_attributes = custom_attributes

    @classmethod
    def from_xml(cls, root, orientation_transform: np.ndarray | None = None):
        """Load a shape from an XML shape node. Used internally for reading LMD generated XML files.

        Args:
            root: XML input node.
        """
        name = root.tag
        well = None
        custom_attributes = {}

        # get number of points
        point_count = int(root.find("PointCount").text)
        points = np.empty((point_count, 2), dtype=int)

        # compile regex
        xpattern = re.compile(r"X_(\d+)")
        ypattern = re.compile(r"Y_(\d+)")

        # parse all points
        for child in root:
            xmatch = re.findall(xpattern, child.tag)
            ymatch = re.findall(ypattern, child.tag)

            if xmatch:
                point_id = int(xmatch[0]) - 1
                points[point_id, 0] = int(child.text)
            elif ymatch:
                point_id = int(ymatch[0]) - 1
                points[point_id, 1] = int(child.text)
            elif child.tag == "CapID":
                well = str(child.text)
            else:
                if child.tag in custom_attributes:
                    warnings.warn(
                        f"Shape attribute {child.tag} already found in shape, overwrite",
                        stacklevel=1,
                    )
                custom_attributes[child.tag] = child.text

        points = np.array(points)

        return cls(
            points=points,
            name=name,
            well=well,
            orientation_transform=orientation_transform,
            **custom_attributes,
        )

    def to_xml(
        self,
        id: int,
        orientation_transform: np.ndarray,
        scale: float,
        *,
        write_custom_attributes: bool = True,
    ):
        """Generate XML shape node needed internally for export.

        Args:
            id: Sequential identifier of the shape as used in the LMD XML format.

            orientation_transform (np.array): Pass orientation_transform which is used if no local orientation transform is set.

            scale (float): Scalling factor used to enable higher decimal precision.

            write_custom_attributes: Write custom attributes to xml file

        Note:
            If the Shape has a custom orientation_transform defined, the custom orientation_transform is applied at this point. If not, the oritenation_transform passed by the parent Collection is used. This highlights an important difference between the Shape and Collection class. The Collection will always has an orientation transform defined and will use `np.eye(2)` by default. The Shape object can have a orientation_transform but can also be set to `None` to use the Collection value.

        """

        # Apply orientation transform. If the Shape has a custom orientation_transform defined, the custom orientation_transform is applied at this point. If not, the oritenation_transform passed by the parent Collection is used. This highlights an important difference between the Shape and Collection class. The Collection will always has an orientation transform defined and will use `np.eye(2)` by default. The Shape object can have a orientation_transform but can also be set to `None` to use the Collection value.

        if self.orientation_transform is not None:
            transformed_points = self.points @ self.orientation_transform * scale
        else:
            transformed_points = self.points @ orientation_transform * scale

        shape = ET.Element(f"Shape_{id}")

        point_count = ET.SubElement(shape, "PointCount")
        point_count.text = f"{len(transformed_points)}"

        if self.well is not None:
            cap_id = ET.SubElement(shape, "CapID")
            cap_id.text = self.well

        if write_custom_attributes:
            for attribute_name, attribute_value in self.custom_attributes.items():
                custom_attribute = ET.SubElement(shape, attribute_name)
                # xml only accepts string values
                custom_attribute.text = str(attribute_value)

        # write points
        for i, point in enumerate(transformed_points):
            id = i + 1
            x = ET.SubElement(shape, f"X_{id}")
            x.text = f"{np.floor(point[0]).astype(int)}"

            y = ET.SubElement(shape, f"Y_{id}")
            y.text = f"{np.floor(point[1]).astype(int)}"

        return shape

    def get_shape_annotation(self, name: str) -> Any | None:
        """Retrieve the value of an attribute from either instance attributes
         or custom attributes by name.

         Searches for the attribute by name in the 1) instance attributes
         2) custom attributes, or 3) issues a warning and returns None

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any | None: The value of the attribute if found, otherwise None.
        """
        if name in self.__dict__:
            return getattr(self, name)
        elif name in self.custom_attributes:
            return self.custom_attributes.get(name)
        else:
            warnings.warn(f"Attribute {name} not found in shape attributes. Returning None.", stacklevel=2)
            return None

    def to_shapely(self):
        return shapely.Polygon(self.points)


class Collection:
    """Class which is used for creating shape collections for the Leica LMD6 & 7. Contains a coordinate system defined by calibration points and a collection of various shapes.

    Args:
        calibration_points: Calibration coordinates in the form of :math:`(3, 2)`.
        orientation_transform: defines transformations performed on the provided coordinate system prior to export as XML. Defaults to the identity matrix.

    Attributes:
        shapes (List[Shape]): Contains all shapes which are part of the collection.
        calibration_points (Optional[np.ndarray]): Calibration coordinates in the form of :math:`(3, 2)`.
        orientation_transform (np.ndarray): defines transformations performed on the provided coordinate system prior to export as XML. This orientation_transform is always applied to shapes when there is no individual orientation_transform provided.
    """

    def __init__(
        self,
        calibration_points: np.ndarray | None = None,
        orientation_transform: np.ndarray | None = None,
        scale: float = 100,
    ):
        self.shapes: list[Shape] = []

        self.calibration_points: np.ndarray | None = calibration_points

        if orientation_transform is None:
            orientation_transform = np.eye(2)  # assign default value

        self.orientation_transform: np.ndarray = orientation_transform

        self.scale: float = scale

        self.global_coordinates = 1

    def stats(self):
        """Print statistics about the Collection in the form of:

        .. code-block::

            ===== Collection Stats =====
            Number of shapes: 208
            Number of vertices: 126,812
            ============================
            Mean vertices: 609.67
            Min vertices: 220.00
            5% percentile vertices: 380.20
            Median vertices: 594.00
            95% percentile vertices: 893.20
            Max vertices: 1,300.00

        """
        lengths = np.array([len(shape.points) for shape in self.shapes])

        num_shapes = len(self.shapes)
        num_vertices = np.sum(lengths)

        median_dp = np.median(lengths).astype(float)
        mean_dp = np.mean(lengths).astype(float)
        max_dp = np.max(lengths).astype(float)
        min_dp = np.min(lengths).astype(float)
        percentile_5 = np.percentile(lengths, 5).astype(float)
        percentile_95 = np.percentile(lengths, 95).astype(float)

        print("===== Collection Stats =====")
        print(f"Number of shapes: {num_shapes:,}")
        print(f"Number of vertices: {num_vertices:,}")
        print("============================")
        print(f"Mean vertices: {mean_dp:,.0f}")
        print(f"Min vertices: {min_dp:,.0f}")
        print(f"5% percentile vertices: {percentile_5:,.0f}")
        print(f"Median vertices: {median_dp:,.0f}")
        print(f"95% percentile vertices: {percentile_95:,.0f}")
        print(f"Max vertices: {max_dp:,.0f}")

    def plot(
        self,
        calibration: bool = True,
        mode: str = "line",
        fig_size: tuple = (5, 5),
        apply_orientation_transform: bool = True,
        apply_scale: bool = False,
        save_name: str | None = None,
        return_fig: bool = False,
        **kwargs,
    ):
        """This function can be used to plot all shapes of the corresponding shape collection.

        Args:
            calibration: Controls wether the calibration points should be plotted as crosshairs. Deactivating the crosshairs will result in the size of the canvas adapting to the shapes. Can be especially usefull for small shapes or debugging.

            fig_size: Defaults to :math:`(10, 10)` Controls the size of the matplotlib figure. See `matplotlib documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_ for more information.

            apply_orientation_transform: Define wether the orientation transform should be applied before plotting.

            save_name (Optional[str], default: None): Specify a filename  for saving the generated figure. By default `None` is provided which will not save a figure.
        """

        modes = ["line", "dots"]

        # Check if Collection scale should be applied or not
        if apply_scale:
            scale = self.scale
        else:
            scale = 1

        if mode not in modes:
            raise ValueError("Mode not known. Please use one of the following plotting modes: line, dots")

        # close current figures
        plt.clf()
        plt.cla()
        plt.close("all")

        fig, ax = plt.subplots(figsize=fig_size, **kwargs)

        # Plot calibration points
        if calibration and self.calibration_points is not None:
            # Apply orientation transform as default behavior
            if apply_orientation_transform:
                calibration_points = self.calibration_points @ self.orientation_transform * scale
            else:
                calibration_points = self.calibration_points * scale

            plt.scatter(calibration_points[:, 0], calibration_points[:, 1], marker="x")

        for shape in self.shapes:
            # Apply orientation transform as default behavior
            if apply_orientation_transform:
                # Use local transform if defined, else use Collection transform
                if shape.orientation_transform is not None:
                    points = shape.points @ shape.orientation_transform * scale
                else:
                    points = shape.points @ self.orientation_transform * scale
            else:
                points = shape.points * scale

            if mode == "line":
                ax.plot(points[:, 0], points[:, 1])

            elif mode == "dots":
                ax.scatter(points[:, 0], points[:, 1], s=10)

        ax.grid(True)
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)

        if return_fig:
            return fig

        plt.show()

    def add_shape(self, shape: Shape):
        """Add a new shape to the collection.

        Args:
            shape: Shape which should be added.
        """

        if isinstance(shape, Shape):
            self.shapes.append(shape)
        else:
            raise TypeError("Provided shape is not of type Shape")

    def new_shape(
        self,
        points: np.ndarray,
        well: str | None = None,
        name: str | None = None,
        **custom_attributes,
    ):
        """Directly create a new Shape in the current collection.

        Args:
            points: Array or list of lists in the shape of `(N,2)`. Contains the points of the polygon forming a shape.

            well: Well in which to sort the shape after cutting. For example A1, A2 or B3.

            name: Name of the shape.

            custom_attributes: Custom shape metadata that can be added as additional xml-element to the shape.
                All values are converted to strings.

        """
        to_add = Shape(
            points,
            well=well,
            name=name,
            orientation_transform=self.orientation_transform,
            **custom_attributes,
        )
        self.add_shape(to_add)

    def join(self, collection: Collection, update_orientation_transform: bool = True):
        """Join the collection with the shapes of a different collection. The calibration markers of the current collection are kept. Please keep in mind that coordinate systems and calibration points must be compatible for correct joining of collections.

        Args:
            collection: Collection which should be joined with the current collection object.
            orientation_transform: If set to True, the orientation transform of the joined collection will be updated to the current collection. If set to False, the orientation transform of the joined collection will not be updated.

        Returns:
            returns self
        """
        if not np.all(self.orientation_transform == collection.orientation_transform):
            if update_orientation_transform:
                shapes = collection.shapes
                for shape in shapes:
                    shape.orientation_transform = self.orientation_transform
            else:
                warnings.warn(
                    "Orientation transform of the joined collection is not equal to the current collection, but update_orientation_transform is set to False. Shapes will be merged without updating the orientation transform.",
                    stacklevel=2,
                )
        self.shapes += collection.shapes

        return self

    def to_geopandas(self, *attrs: str) -> gpd.GeoDataFrame:
        """Return geopandas dataframe of collection

        Args:
            *attrs (str): Optional attributes of the shapes in the collection to be added as metadata columns

        Returns:
            geopandas.GeoDataFrame: Representation of all shapes and optional metadata

        Example:
        .. code-block:: python
            # Generate collection
            collection = pylmd.Collection()
            shape = pylmd.Shape(
                    np.array([[ 0,  0], [ 0, -1], [ 1,  0], [ 0,  0]]),
                    well="A1",
                    name="Shape_1",
                    metadata1="A",
                    metadata2="B",
                    orientation_transform=None
                )
            collection.add_shape(shape)

            # Get geopandas object
            collection.to_geopandas()
            >       geometry
                0   POLYGON ((0 0, 0 -1, 1 0, 0 0))

            collection.to_geopandas("well", "name", "metadata1", "metadata2")
            >       well    name            metadata1 metadata2  geometry
                0   A1      Shape_1         A         B          POLYGON ((0 0, 0 -1, 1 0, 0 0))
        """
        metadata = (
            pd.DataFrame(
                [[shape.get_shape_annotation(att) for att in attrs] for shape in self.shapes],
                columns=attrs,
            )
            if (attrs is not None)
            else None
        )
        geometry = [shape.to_shapely() for shape in self.shapes]

        return gpd.GeoDataFrame(data=metadata, geometry=geometry)

    # load xml from file
    def load(self, file_location: str, *, raise_shape_errors: bool = False):
        """Can be used to load a shape file from XML. Both, XMLs generated with py-lmd and the Leica software can be used.
        Args:
            file_location: File path pointing to the XML file.
            raise_errors: Whether to raise errors during shape collection. If `False` raises a warning.

        """

        tree = ET.parse(file_location)
        root = tree.getroot()

        cal_point_len = 0

        # count calibration points
        for child in root:
            if "CalibrationPoint" in child.tag:
                cal_point_len += 1

        self.calibration_points = np.ones((cal_point_len // 2, 2), dtype=int)

        for child in root:
            if child.tag == "GlobalCoordinates":
                self.global_coordinates = int(child.text)

            # Load calibration points
            elif "CalibrationPoint" in child.tag:
                axes = child.tag[0]
                axes_id = 0 if axes == "X" else 1
                shape_id = int(child.tag[-1]) - 1
                value = int(child.text)

                self.calibration_points[shape_id, axes_id] = value

            # Load shapes
            elif "Shape_" in child.tag:
                try:
                    new_shape = Shape.from_xml(child)
                    self.shapes.append(new_shape)

                except ValueError as e:
                    if raise_shape_errors:
                        raise ValueError(e) from e
                    else:
                        warnings.warn(str(e), stacklevel=1)
                    continue

    def load_geopandas(
        self,
        gdf: gpd.GeoDataFrame,
        geometry_column: str = "geometry",
        name_column: str | None = None,
        well_column: str | None = None,
        calibration_points: np.ndarray | None = None,
        global_coordinates: int | None = None,
        custom_attribute_columns: str | list[str] | None = None,
    ) -> None:
        """Create collection from a geopandas dataframe

        Args:
            gdf (geopandas.GeoDataFrame): Collection of shapes and optional metadata
            geometry_column (str, default: geometry): Name of column storing Shapes as `shapely.Polygon`, defaults to geometry
            well_column (str, optional): Column storing of well id as additional metadata
            calibration_points (np.ndarray, optional): Calibration points of collection
            global_coordinates (int, optional): Number of global coordinates
            custom_attribute_columns Custom shape metadata that will be added as additional xml-element to the shape.
                Can be column name, list of column names or None

        Example:

        ..  code-block:: python

            from lmd.lib import Collection
            import geopandas as gpd
            import shapely

            gdf = gpd.GeoDataFrame(
                data={"well": ["A1"], "name": ["test"]}, geometry=[shapely.Polygon([[0, 0], [0, 1], [1, 0], [0, 0]])]
            )

            # Create collection
            c = Collection()

            # Export well metadata
            c.load_geopandas(gdf, well_column="well")
            assert c.to_geopandas("well").equals(gdf)

            # Do not export well metadata
            c.load_geopandas(gdf)
            assert c.to_geopandas().equals(gdf.drop(columns="well"))
        """
        # Update attributes
        if calibration_points is not None:
            self.calibration_points = calibration_points
        if global_coordinates is not None:
            self.global_coordinates = global_coordinates

        if custom_attribute_columns is None:
            custom_attribute_columns = []
        if isinstance(custom_attribute_columns, str):
            custom_attribute_columns = [custom_attribute_columns]

        self.shapes = [
            Shape(
                points=np.array(row[geometry_column].exterior.coords),
                name=row[name_column] if name_column is not None else None,
                well=row[well_column] if well_column is not None else None,
                **{att: row[att] for att in custom_attribute_columns},
            )
            for _, row in gdf.iterrows()
        ]

    # save xml to file
    def save(self, file_location: str, encoding: str = "utf-8"):
        """Can be used to save the shape collection as XML file.

        file_location: File path pointing to the XML file.
        """

        root = ET.Element("ImageData")

        # write global coordinates
        global_coordinates = ET.SubElement(root, "GlobalCoordinates")
        global_coordinates.text = "1"

        # transform calibration points
        transformed_calibration_points = self.calibration_points @ self.orientation_transform * self.scale

        # write calibration points
        for i, point in enumerate(transformed_calibration_points):
            print(point)

            id = i + 1
            x = ET.SubElement(root, f"X_CalibrationPoint_{id}")
            x.text = f"{np.floor(point[0]).astype(int)}"

            y = ET.SubElement(root, f"Y_CalibrationPoint_{id}")
            y.text = f"{np.floor(point[1]).astype(int)}"

        # write shape length
        shape_count = ET.SubElement(root, "ShapeCount")
        shape_count.text = f"{len(self.shapes)}"

        # write shapes
        for i, shape in enumerate(self.shapes):
            id = i + 1

            # apply Collection orientation_transform and scale
            root.append(shape.to_xml(id, self.orientation_transform, self.scale))

        # write root
        tree = ET.ElementTree(element=root)
        tree.write(file_location, encoding="utf-8", xml_declaration=True, pretty_print=True)

    def svg_to_lmd(
        self,
        file_location,
        offset=None,
        divisor=3,
        multiplier=60,
        rotation_matrix=np.eye(2),
        orientation_transform=None,
    ):
        """Can be used to save the shape collection as XML file.

        Args:
            file_location: File path pointing to the SVG file.

            orientation_transform: Will superseed the global transform of the Collection.

            rotation_matrix:

        """

        if offset is None:
            offset = [0, 0]
        orientation_transform = self.orientation_transform if orientation_transform is None else orientation_transform

        svg = SVG.parse(file_location)
        list(svg.elements())

        for path in svg:
            pl = []
            n_points = int(path.length() // divisor)
            linspace = np.linspace(0, 1, n_points)

            for index in linspace:
                poly = np.array(path.point(index))
                pl.append([poly[0], -poly[1]])

            arr = np.array(pl) @ rotation_matrix * multiplier + offset

            to_add = Shape(points=arr, orientation_transform=orientation_transform)
            self.add_shape(to_add)
