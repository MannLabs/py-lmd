from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from lxml import etree as ET
from matplotlib import image
from skimage import data, color
from xml.dom import minidom
import matplotlib.ticker as ticker
from svgelements import SVG




class Collection:
    """Class which is used for creating shape collections for the Leica LMD6 & 7. Contains a coordinate system defined by calibration points and a collection of various shapes.
    """
    
    def __init__(self, calibration_points: Optional[np.ndarray] = None):
        """Class which is used for creating shape collections for the Leica LMD6 & 7. Contains a coordinate system defined by calibration points and a collection of various shapes.
        
        Args:
            calibration_points: Calibration coordinates in the form of :math:`(3, 2)`.
            
        Attributes:
            shapes: Contains all shapes which are part of the collection.
            
            calibration_points: Calibration coordinates in the form of :math:`(3, 2).
            
            orientation_transform: defines transformations performed on the provided coordinate system prior to export as XML. This orientation_transform is always applied to shapes when there is no individual orienation_transform provided.
        """
        
        self.shapes: List[LMD_shape] = []
        
        self.calibration_points: Optional[np.ndarray] = calibration_points
            
        
        self.orientation_transform: np.ndarray = np.eye(2)
            
        self.scale = 100
        
        self.global_coordinates = 1
        
    def plot(self, calibration: bool = True, 
             mode: str = "line", 
             fig_size: tuple = (10,10),
            apply_orientation_transform: bool = True):
        
        """This function can be used to plot all shapes of the corresponding shape collection.
        
        Args:
            calibration: Controls wether the calibration points should be plotted as crosshairs. Deactivating the crosshairs will result in the size of the canvas adapting to the shapes. Can be especially usefull for small shapes or debugging.

            fig_size: Defaults to :math:`(10, 10)` Controls the size of the matplotlib figure. See `matplotlib documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_ for more information.
            
            apply_orientation_transform: Define wether the orientation transform should be applied before plotting.
        """
        
        modes = ["line","dots"]
        
        if not mode in modes:
            raise ValueError("mode not known")
        # check for calibration points
        
        cal = np.array(self.calibration_points).T


        plt.clf()
        plt.cla()
        plt.close("all")

        fig, ax = plt.subplots(figsize=fig_size)
        
        # Plot calibration points
        if calibration and self.calibration_points is not None:
            
            # Apply orientation transform as default behavior
            if apply_orientation_transform:
                calibration = self.calibration_points @ self.orientation_transform
            else:
                calibration = self.calibration_points
                
            plt.scatter(calibration[:,0],calibration[:,1],marker="x")
            
        for shape in self.shapes:
            
            # Apply orientation transform as default behavior
            if apply_orientation_transform:
                points = shape.points @ shape.orientation_transform
            else:
                points = shape.points
     
            
            if mode == "line":
                ax.plot(points[:,0],points[:,1])           
                
            elif mode == "dots":
                ax.scatter(points[:,0],points[:,1], s=10)
            
        

        ax.grid(True)
        ax.ticklabel_format(useOffset=False)
        plt.axis('equal')
        plt.show()
        
    def add_shape(self, shape: Shape):
        """Add a new shape to the collection.
        
        Args:
            shape: Shape which should be added.
        """
        
        if isinstance(shape, Shape):
            self.shapes.append(shape)
        else:
            TypeError('Provided shape is not of type Shape')
            
        
    def new_shape(self, points: np.ndarray, 
                 well: Optional[str] = None, 
                 name: Optional[str] = None):
        
        """Directly create a new Shape in the current collection.

        Args:
            points: Array or list of lists in the shape of `(N,2)`. Contains the points of the polygon forming a shape.
            
            well: Well in which to sort the shape after cutting. For example A1, A2 or B3.
            
            name: Name of the shape.
        """

        to_add = Shape(points, well=well, name=name, orientation_transform = self.orientation_transform)
        self.add_shape(to_add)
    
    def join(self,  collection: Collection):
        """Join the collection with the shapes of a different collection. The calibration markers of the current collection are kept. Please keep in mind that coordinate systems and calibration points must be compatible for correct joining of collections.
        
        Args:
            collection: Collection which should be joined with the current collection object.
            
        """
        self.shapes += collection.shapes
        
    # load xml from file
    def load(self, file_location: str):
        """Can be used to load a shape file from XML. Both, XMLs generated with py-lmd and the Leica software can be used.
        Args:
            file_location: File path pointing to the XML file.

        """

        tree = ET.parse(file_location)
        root = tree.getroot()        
        
        cal_point_len = 0
        
        # count calibration points
        for child in root:
            if "CalibrationPoint" in child.tag:
                cal_point_len += 1
                
        self.calibration_points = np.ones((cal_point_len//2,2),dtype=int)
        
        
        for child in root:
            
            if child.tag == "GlobalCoordinates":
                self.global_coordinates = int(child.text)
                
            # Load calibration points
            elif "CalibrationPoint" in child.tag:
                axes = child.tag[0]
                axes_id = 0 if axes == "X" else 1
                shape_id = int(child.tag[-1])-1
                value = int(child.text)
                
                self.calibration_points[shape_id,axes_id] = value
            
            
            # Load shapes
            elif "Shape_" in child.tag:  
                new_shape = LMD_shape()
                new_shape.from_xml(child)
                self.shapes.append(new_shape)
                
    #save xml to file            
    def save(self, file_location: str, encoding: str = 'utf-8'):
        """Can be used to save the shape collection as XML file.

        file_location: File path pointing to the XML file.
        """
       
        root = ET.Element("ImageData")
        
        # write global coordinates
        global_coordinates = ET.SubElement(root, 'GlobalCoordinates')
        global_coordinates.text = "1"
        
        # write calibration points
        for i, point in enumerate(self.calibration_points):
            id = i +1
            x = ET.SubElement(root, 'X_CalibrationPoint_{}'.format(id))
            x.text = "{}".format(point[0])
            
            y = ET.SubElement(root, 'Y_CalibrationPoint_{}'.format(id))
            y.text = "{}".format(point[1])
            
        # write shape length
        shape_count = ET.SubElement(root, 'ShapeCount')
        shape_count.text = "{}".format(len(self.shapes))
        
        # write shapes
        for i, shape in enumerate(self.shapes):
            id = i +1
            root.append(shape.to_xml(id))
        
        # write root 
        tree = ET.ElementTree(element=root)
        tree.write(file_location, 
                   encoding="utf-8", 
                   xml_declaration=True, 
                   pretty_print=True)
        
    def svg_to_lmd(self, file_location, 
                   offset=[0,0], 
                   divisor=3, 
                   multiplier=60, 
                   rotation_matrix = np.eye(2), 
                   orientation_transform = None):
        """Can be used to save the shape collection as XML file.
        
        Args: 
            file_location: File path pointing to the SVG file.
            
            orientation_transform: Will superseed the global transform of the Collection.
            
            rotation_matrix: 
            
        """
        
        orientation_transform = self.orientation_transform if orientation_transform is None else orientation_transform
        
        svg = SVG.parse(file_location)
        paths = list(svg.elements())

        poly_list = []
        for path in svg:

            pl = []
            n_points = int(path.length() // divisor)
            linspace = np.linspace(0,1,n_points)

            for index in linspace:
                poly = np.array(path.point(index))
                pl.append([poly[0],-poly[1]])

            arr = np.array(pl)@rotation_matrix * multiplier+offset
            to_add = Shape(points=arr, orientation_transform = orientation_transform)
            self.add_shape(to_add) 
        
        
class Shape:
    """Class for creating a single shape object."""

    def __init__(self, points: np.ndarray, 
                 well: Optional[str] = None, 
                 name: Optional[str] = None,
                orientation_transform = np.eye(2)):
        
        """Class for creating a single shape.
        
        Args:
            points: Array or list of lists in the shape of `(N,2)`. Contains the points of the polygon forming a shape.
            
            well: Well in which to sort the shape after cutting. For example A1, A2 or B3.
            
            name: Name of the shape.
        """
        
        # Orientation transform of shapes
        self.orientation_transform: Optional[np.ndarray] = np.eye(2)
            
        # Allthoug a numpy array is recommended, list of lists is accepted
        points = np.array(points)
        
        # Assert correct dimensions
        point_shapes = points.shape
        if ((len(point_shapes) != 2) or 
            (point_shapes[1] != 2) or
            (point_shapes[0] == 0)):
            raise ValueError('please provide a numpy array of shape (N, 2)')
            
        self.points: np.ndarray = points
            
        self.name: Optional[str] = name
        self.well: Optional[str] = well
            
    def from_xml(self, root):
        """Load a shape from an XML shape node. Used internally for reading LMD generated XML files.
        
        Args:
            root: XML input node.
        """
        self.name = root.tag

        # get number of points
        point_count = int(root.find("PointCount").text)   
        self.points = np.ones((point_count,2),dtype=int)
        
        # parse all points
        for child in root:
            if "_" in child.tag: 
                
                tokens = child.tag.split("_")
                
                axes = tokens[0]
                axes_id = 0 if axes == "X" else 1
                
                point_id = int(tokens[-1])-1
                value = int(child.text)

                self.points[point_id,axes_id] = value
                
        self.points = np.array(self.points)
    
    def to_xml(self, id: int):
        """Generate XML shape node needed internally for export.
        
        Args:
            id: Sequential identifier of the shape as used in the LMD XML format.
        """

        shape = ET.Element("Shape_{}".format(id))
        
        point_count = ET.SubElement(shape, "PointCount")
        point_count.text = "{}".format(len(self.points))
        
        if self.well is not None:
            cap_id = ET.SubElement(shape, "CapID")
            cap_id.text = self.well
        
        
        # write points
        for i, point in enumerate(self.points):
            id = i +1
            x = ET.SubElement(shape, 'X_{}'.format(id))
            x.text = "{}".format(point[0])
            
            y = ET.SubElement(shape, 'Y_{}'.format(id))
            y.text = "{}".format(point[1])
        
        return shape
        

class LMD_object:

    """Class which is used for creating shape collections for the Leica LMD6 & 7. Contains a coordinate systeme defined by calibration points and a collection of various shapes."""
        
    
    def __init__(self, calibration_points = []):
        """
        :param calibration_points: Calibration coordinates in the form of :math:`(3, 2)`.
        :type calibration_points: :class:`numpy.array`, optional
        """
        

        self.shapes = []
        self.calibration_points = calibration_points
        self.global_coordinates = 1
    
    
    def plot(self, calibration=True, mode="line", fig_size=(10,10)):
        """This function can be used to plot all shapes of the corresponding :class:`lmd.LMD-object`.

        :param calibration: Controls wether the calibration points should be plotted as crosshairs. Deactivating the crosshairs will result in the size of the canvas adapting to the shapes. Can be especially usefull for small shapes or debugging.
        :type calibration: boolean, optional, defaults to True

        :param fig_size: Controls the size of the matplotlib figure. See `matplotlib documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_ for more information. 
        :type fig_size: (float, float), optional, defaults to :math:`(10, 10)`
        """
        modes = ["line","dots"]
        
        if not mode in modes:
            raise ValueError("mode not known")
        # check for calibration points
        
        cal = np.array(self.calibration_points).T


        plt.clf()
        plt.cla()
        plt.close("all")

        fig, ax = plt.subplots(figsize=fig_size)
        #ax.invert_yaxis()
        if calibration and len(self.calibration_points) > 0:
            plt.scatter(cal[1],cal[0],marker="x")
            
        if mode == "line":
            for shape in self.shapes:
                ax.plot(shape.points.T[1],shape.points.T[0])
        elif mode == "dots":
            for shape in self.shapes:
                ax.scatter(shape.points.T[1],shape.points.T[0], s=10)
            
        

        ax.grid(True)

        ax.ticklabel_format(useOffset=False)
        plt.axis('equal')
        plt.show()
       
             
            
    def add_shape(self, shape):
        """Add a new shape to the current collection.

        :param shape: Shape to append.
        :type shape: :class:`lmd.LMD_shape`
        """

        self.shapes.append(shape)
        
    def new_shape(self, points, well = None):
        """Directly create a new Shape in the current collection.

        :param points: Array or list of lists in the shape of :math:`(N,2)`. Should contain the coordinates of the shape in the defined coordinate system.
        :type points: :class:`numpy.array`, optional

        :param well: Well in which to sort the shape after cutting. For example A1, A2, B3.
        :type well: string, optional
        """

        to_add = LMD_shape(points=points, well=well)
        self.add_shape(to_add)
        
    
    # load xml from file
    def load(self, file_location):
        """Can be used to load a shape file from XML. Both, XMLs generated with py-lmd and the Leica software can be used.

        :param file_location: File path pointing to the XML file.
        :type file_location: string
        """

        tree = ET.parse(file_location)
        root = tree.getroot()        
        
        cal_point_len = 0
        
        # count calibration points
        for child in root:
            if "CalibrationPoint" in child.tag:
                cal_point_len += 1
                
        self.calibration_points = np.ones((cal_point_len//2,2),dtype=int)
        
        
        for child in root:
            
            if child.tag == "GlobalCoordinates":
                self.global_coordinates = int(child.text)
                
            elif "CalibrationPoint" in child.tag:
                axes = child.tag[0]
                axes_id = 0 if axes == "X" else 1
                shape_id = int(child.tag[-1])-1
                value = int(child.text)
                
                self.calibration_points[shape_id,axes_id] = value
                
            elif "Shape_" in child.tag:  
                new_shape = LMD_shape()
                new_shape.from_xml(child)
                self.shapes.append(new_shape)
                
    #save xml to file            
    def save(self, filename):
        """Can be used to save the shape collection as XML file.

        :param filename: File path pointing to the XML file.
        :type filename: string
        """
       
        root = ET.Element("ImageData")
        
        # write global coordinates
        global_coordinates = ET.SubElement(root, 'GlobalCoordinates')
        global_coordinates.text = "1"
        
        # write calibration points
        for i, point in enumerate(self.calibration_points):
            id = i +1
            x = ET.SubElement(root, 'X_CalibrationPoint_{}'.format(id))
            x.text = "{}".format(point[0])
            
            y = ET.SubElement(root, 'Y_CalibrationPoint_{}'.format(id))
            y.text = "{}".format(point[1])
            
        # write shape length
        shape_count = ET.SubElement(root, 'ShapeCount')
        shape_count.text = "{}".format(len(self.shapes))
        
        # write shapes
        for i, shape in enumerate(self.shapes):
            id = i +1
            root.append(shape.to_xml(id))
        
        tree = ET.ElementTree(element=root)
        tree.write(filename, encoding="utf-8", xml_declaration=True, pretty_print=True)
        #tree.write(filename, pretty_print=True)
        
        #xmlstr = "\n".join(ET.tostringlist(root,encoding="unicode"))
        #with open(filename, "w") as f:
            #f.write(xmlstr)
            
    def image_to_lmd(self, path, offset=[0,0]):
        img = image.imread(path)
        img = color.rgb2gray(img)

        size = img.shape
        print(size)

        obj = LMD_object()
        obj.calibration_points = [(-10,-10),(110,0),(110,110)]

        for y,row in enumerate(img):

            for x,val in enumerate(row):
                if val == 0:
                    xv = 200 * x + offset[0]
                    yv = 200 * y + offset[1]

                    d=160

                    arr = np.array([[xv,yv],[xv+d,yv],[xv+d,yv+d],[xv,yv+d],[xv,yv]])


                    to_add = LMD_shape(points=arr)
                    self.add_shape(to_add)    
                    
    def svg_to_lmd(self, path, offset=[0,0], divisor=3, multiplier=60, rotation= np.eye(2), orientation_transform= np.eye(2)):
        """Can be used to save the shape collection as XML file.

        :param filename: File path pointing to the XML file.
        :type filename: string
        """
        
        svg = SVG.parse(path)
        paths = list(svg.elements())

        poly_list = []
        for path in svg:

            pl = []
            n_points = int(path.length() // divisor)
            linspace = np.linspace(0,1,n_points)

            for index in linspace:
                poly = np.array(path.point(index))
                pl.append([-poly[1],poly[0]])

            arr = np.array(pl)@rotation * multiplier+offset
            arr = arr @ orientation_transform
            to_add = LMD_shape(points=arr)
            self.add_shape(to_add) 
            
    def join(self,  lmd_object_to_join):
        
        self.shapes += lmd_object_to_join.shapes
        
            
class LMD_shape:
    """Class for creating a single shape object."""
    
    
    def __init__(self, points=[], well=None, name=""):
        """Class for creating a single shape object.
        
        :param points: Array or list of lists in the shape of (N,2). Should contain the coordinates of the shape in the defined coordinate system.
        :type points: :class:`numpy.array`, optional

        :param well: Well in which to sort the shape after cutting. For example A1, A2, B3.
        :type well: string, optional

        :param name: Name of the shape.
        :type name: string, optional
        """
        self.name = ""
        self.points = np.array(points, dtype=int)
        self.well = well
        
    def from_xml(self, root):
        """Load a shape from an XML shape node. Used internally for reading LMD generated XML files.
        
        :param root: XML input node.
        :type root: :class:`lxml.etree.Element`
        """
        self.name = root.tag

        # get number of points
        point_count = int(root.find("PointCount").text)   
        self.points = np.ones((point_count,2),dtype=int)
        
        # parse all points
        for child in root:
            if "_" in child.tag: 
                
                tokens = child.tag.split("_")
                
                axes = tokens[0]
                axes_id = 0 if axes == "X" else 1
                
                point_id = int(tokens[-1])-1
                value = int(child.text)

                self.points[point_id,axes_id] = value
                
        self.points = np.array(self.points)
    
    def to_xml(self, id):
        """Generate XML shape node needed internally for export.
        
        :param id: Sequential identifier of the shape as used in the LMD XML format.
        :type id: int
        """

        shape = ET.Element("Shape_{}".format(id))
        
        point_count = ET.SubElement(shape, "PointCount")
        point_count.text = "{}".format(len(self.points))
        
        if self.well is not None:
            cap_id = ET.SubElement(shape, "CapID")
            cap_id.text = self.well
        
        
        # write points
        for i, point in enumerate(self.points):
            id = i +1
            x = ET.SubElement(shape, 'X_{}'.format(id))
            x.text = "{}".format(point[0])
            
            y = ET.SubElement(shape, 'Y_{}'.format(id))
            y.text = "{}".format(point[1])
        
        return shape
