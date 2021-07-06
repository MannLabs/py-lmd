import numpy as np
import matplotlib.pyplot as plt
from lxml import etree as ET
from matplotlib import image
from skimage import data, color
from xml.dom import minidom
import matplotlib.ticker as ticker
from svgelements import SVG

class LMD_object:
    
    
    
    def __init__(self):
        """This function takes a preprocessed image with low background noise and extracts and segments the cells.
    	Extraction is performed based on global thresholding, therefore a preprocessed image with homogenous low noise background is needed.
    
    	:param image: 2D numpy array of type float containing the image. 
    	:type image: class:`numpy.array`
    
    	"""
        self.shapes = []
        self.calibration_points = []
        self.global_coordinates = 1
    
    
    def plot(self, calibration=True, fig_size=(10,10)):
        # check for calibration points
        if len(self.calibration_points) > 0:
            cal = np.array(self.calibration_points).T
            

            plt.clf()
            plt.cla()
            plt.close("all")

            fig, ax = plt.subplots(figsize=fig_size)
            #ax.invert_yaxis()
            if calibration:
                plt.scatter(cal[1],cal[0],marker="x")
            
            for shape in self.shapes:
                
                ax.plot(shape.points.T[1],shape.points.T[0])
                
            ax.grid(True)
            
            ax.ticklabel_format(useOffset=False)
            plt.show()
        else:
            print("please define calibration points before plotting")
             
            
    def add_shape(self, shape):
        self.shapes.append(shape)
        
    def new_shape(self, points, well = None):
        to_add = LMD_shape(points=points, well=well)
        self.add_shape(to_add)
        

            
    
    # load xml from file
    def load(self, file_location):
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
                    
    def svg_to_lmd(self, path, offset=[0,0], divisor=3, multiplier=60):
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

            arr = np.array(pl)*multiplier+offset
            to_add = LMD_shape(points=arr)
            self.add_shape(to_add) 
        
            
class LMD_shape:
    
    
    
    def __init__(self, points=[], well=None):
        self.name = ""
        self.points = np.array(points, dtype=int)
        self.well = well
        
    def from_xml(self, root):
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