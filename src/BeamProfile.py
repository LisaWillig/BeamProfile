import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import sys
import os
from datetime import datetime
from PyQt5 import QtGui, uic
import PyQt5
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from pyqtgraph.Qt import QtGui, QtCore
import time as t
from matplotlib import cm
from scipy.optimize import curve_fit
from pypylon import genicam
import pypylon
import easygui
import matplotlib.pyplot as plt
import lmfit as lm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from modules.BaslerCommunication import BaslerMultiple as Basler
from modules.saveGUISettings import GUISettings
from modules.analysis import *
##### Number of Cams, Name (UserDeviceID) of Cams
CamsToUse = 1
namesCamsToUse = {0:'MOKE_1'}

#### Other Variables
SavingDestination = "C:\\PythonSoftware\\Beamprofile\\SavedData\\"
ConfigFileName = "PositionConfigFile.dat"
conversionFactor = 1 # (mm/px, calculate real world mm on camera from px)
imgColormap = "inferno"
imageCalculations = dict()


class saveClass():
    fn = "BeamStabilization\\"+str(ConfigFileName)

    def createOrOpenConfigFile(self):

        header = False
        file = open(self.fn, 'a+')
        if os.stat(self.fn).st_size == 0:
            header = True
        if header:
            file.write('#Cam0 X\t Cam0 Y\t Cam1 X\t Cam1 Y\n')
        return file

    def readLastConfig(self):
        open(self.fn, 'a+')
        if os.stat(self.fn).st_size != 0:
            try:
                with open(self.fn, 'r') as file:
                    lines = file.read().splitlines()
                    last_line = lines[-1]
                imageCalculations[0]["GoalPixel_X"] = float(last_line.split('\t')[0])
                imageCalculations[0]["GoalPixel_Y"] = float(last_line.split('\t')[1])
                imageCalculations[1]["GoalPixel_X"] = float(last_line.split('\t')[2])
                imageCalculations[1]["GoalPixel_Y"] = float(last_line.split('\t')[3])
                return True
            except FileNotFoundError:
                return False

    def writeNewCenter(self):
        file = self.createOrOpenConfigFile()
        file.write(str(imageCalculations[0]["Center_GaussFitX"]) + '\t')
        file.write(str(imageCalculations[0]["Center_GaussFitY"]) + '\t')
        file.write(str(imageCalculations[1]["Center_GaussFitX"]) + '\t')
        file.write(str(imageCalculations[1]["Center_GaussFitY"]) + '\t')
        file.write('\n')
        file.close()

class Logging():

    def __init__(self):
        self.createFolderAndFile()

    def createFolderAndFile(self):
        if not os.path.exists(SavingDestination+"\\Logging"):
            os.makedirs(SavingDestination+"\\Logging")
        os.chdir(SavingDestination+"\\Logging")
        self.timeStamp = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.file = open(str(self.timeStamp), 'a+')

        self.file.write('# timeStamp\t FWHMX1\t FWHMY1\t FWHMX2\t '
                        'FWHMY2\t CoM_X1\t CoM_X2\t '
                        'CoM_Y1\tCoM_Y2\tGausscenterX1\t '
                        'GausscenterX2\t '
                   'GausscenterY1\t GausscenterY2\n')

    def saveValues(self):

        self.file.write(str(datetime.now().strftime("%Y%m%d_%H%M%S")) + '\t')

        self.file.write(str(imageCalculations[0]["FWHM_X"]) + '\t')
        self.file.write(str(imageCalculations[0]["FWHM_Y"]) + '\t')
        self.file.write(str(imageCalculations[1]["FWHM_X"]) + '\t')
        self.file.write(str(imageCalculations[1]["FWHM_Y"]) + '\t')

        self.file.write(str(imageCalculations[0]["CoM_X"]) + '\t')
        self.file.write(str(imageCalculations[1]["CoM_X"]) + '\t')

        self.file.write(str(imageCalculations[0]["CoM_Y"]) + '\t')
        self.file.write(str(imageCalculations[1]["CoM_Y"]) + '\t')

        self.file.write(str(imageCalculations[0]["Center_GaussFitX"]) + '\t')
        self.file.write(str(imageCalculations[1]["Center_GaussFitX"]) + '\t')

        self.file.write(str(imageCalculations[0]["Center_GaussFitY"]) + '\t')
        self.file.write(str(imageCalculations[1]["Center_GaussFitY"]) + '\n')

    def closeFile(self):

        self.file.close()

class MyWindow(PyQt5.QtWidgets.QMainWindow):

    def __init__(self, parent=None):

        super(MyWindow, self).__init__(parent)
        self.ui = uic.loadUi('GUI\\Beamprofile.ui', self)

        self.ui.setWindowTitle("Beamprofile")
        self.init_ui()
        #self.Save = saveClass()
        #if self.Save.readLastConfig():
        #    self.config = True
        self.initIntegrationTime()
        self.average = 1
        self.bZoom = False
        self.init = 1
        self.bFitX = False
        self.bFitY = False
        self.FList = []
        self.fwhmXList=[]
        self.fwhmYList = []

        self.Setting = GUISettings()
        self.Setting.guirestore(self.ui, QtCore.QSettings('./GUI/saved.ini', QtCore.QSettings.IniFormat))

        self.show()

        self.btn_Exit.clicked.connect(self.close)
        self.line_expTime.returnPressed.connect(self.setIntTime)
        self.line_average.returnPressed.connect(self.setAverage)
        self.slide_expTime.valueChanged.connect(self.printValue)
        self.btn_Zoom.toggled.connect(self.zoomToROI)
        self.btn_Save.clicked.connect(self.saveAnalysis)
        self.check_chopper.toggled.connect(self.chopperText)
        '''
        self.readLastReferencePosition()
        self.btn_Start.clicked.connect(self.startAligning)
        self.btn_setCenter.clicked.connect(self.newCenter)
        self.btn_showCenter.clicked.connect(self.displayCenterLine)
        self.btn_moveMirror.clicked.connect(self.moveSingleMirror)
        '''

        self.Main()

    def chopperText(self):
        if self.check_chopper.isChecked():
            self.check_chopper.setText("with Chopper")
        else:
            self.check_chopper.setText("w/o Chopper")

    @staticmethod
    def createTimeStamp_Date():
        """
        Main folder for data saving is named after Timestamp to ensure
        unique name tha tcannot be overriden accidently
        :return: date
        """

        return str(datetime.now().strftime("%Y%m%d"))

    @staticmethod
    def createTimeStamp_Time():
        """
        Main folder for data saving is named after Timestamp to ensure
        unique name tha tcannot be overriden accidently
        :return: time
        """

        return str(datetime.now().strftime("%H%M%S"))

    def saveAnalysis(self):

        date = self.createTimeStamp_Date()
        path = SavingDestination+date

        if not os.path.exists(path):
            os.makedirs(path)

        name = self.createTimeStamp_Time()

        np.savetxt(path+"\\"+date +"_"+name+"_image.txt", imageCalculations["Image"])

        self.createOrOpenListFile(SavingDestination,name, date)
        self.zoomToROI()
        print(self.size[0])
        print(self.xMin)
        
        minX = float(self.xMin)*float(self.size[0])
        maxX = self.xMax*float(self.size[0])
        minY = self.yMin*float(self.size[1])
        maxY = self.yMax*float(self.size[1])
        
        analysis(imageCalculations["Image"], self.size[0], self.size[1],minX ,maxX ,minY,maxY , path, date, name)
        

    def createOrOpenListFile(self, path, name, date):

        self.fn = path + "\\BeamprofileList.txt"
        header = False
        file = open(self.fn, 'a+')
        if os.stat(self.fn).st_size == 0:
            header = True
        if header:
            file.write('#date\t time\t FWHMX(µm)\t FWHMY(µm)\t FWHMX_Slice(µm)\t FWHMY_Slice(µm)\n')
        file.write(date + "\t")
        file.write(name + "\t")
        file.write(str(np.round(imageCalculations["FWHM_X"],2))+"\t")
        file.write(str(np.round(imageCalculations["FWHM_Y"],2)) + "\t")
        file.write(str(np.round(imageCalculations["SliceFWHM_X"],2)) + "\t")
        file.write(str(np.round(imageCalculations["SliceFWHM_Y"],2)) + "\n")
        file.close()

    def zoomToROI(self):
        self.xMin = imageCalculations["Center_GaussFitX"] - len(imageCalculations["SumX"])/3
        self.xMax = imageCalculations["Center_GaussFitX"] + len(imageCalculations["SumX"])/3

        self.yMin = imageCalculations["Center_GaussFitY"] - len(imageCalculations["SumY"])/3
        self.yMax = imageCalculations["Center_GaussFitY"] + len(imageCalculations["SumY"])/3

        if self.btn_Zoom.isChecked():
            self.bZoom = True
        else:
            self.bZoom = False
            

    def setAverage(self):
        self.average = int(self.line_average.text())

    def initIntegrationTime(self):
        exposureTime = 10000  # µs
        self.initCam(exposureTime)
        self.getIntLimits()
        self.slide_expTime.setMinimum(self.getIntLimits()[0])
        self.slide_expTime.setMaximum(self.getIntLimits()[1])
        sliderPos = self.calcSlider(val=exposureTime*1000)
        self.slide_expTime.setValue(sliderPos)
        self.line_expTime.setText(str(exposureTime / 1000))

    def printValue(self):
        val = np.round(self.calcSlider(), 2)
        val = self.slide_expTime.value()
        self.line_expTime.setText(str(val / 1000))
        self.setIntTime()

    def setIntTime(self):
        self.cam.setIntegrationTime(int(float(self.line_expTime.text())*1000))

    def getIntLimits(self):
         return self.cam.getIntLimits()

    # source: https://gist.github.com/justinfx/3427750
    def calcSlider(self, val=0):
        """
        Just a standard math fit/remap function
            number v 		- initial value from old range
            number oldmin 	- old range min value
            number oldmax 	- old range max value
            number newmin 	- new range min value
            number newmax 	- new range max value
        Example:
            fit(50, 0, 100, 0.0, 1.0)
            # 0.5
        """
        if val == 0:
            val = self.slide_expTime.value()*10
        oldmin = self.slide_expTime.minimum()
        oldmax = self.slide_expTime.maximum()
        newmin = self.getIntLimits()[0]
        newmax = self.getIntLimits()[1]

        scale = ((float(val) - oldmin) / (oldmax - oldmin))
        new_range = scale * (newmax - newmin)
        if newmin < newmax:
            return (newmin + new_range)/1000
        else:
            return (newmin - new_range)/1000

    def init_ui(self):

        # Left Image
        self.vb = self.ImageBox.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.Image = pg.ImageItem()

        # Lines
        self.xCenter = pg.InfiniteLine(pen=(215, 0, 26), angle=90)
        self.yCenter = pg.InfiniteLine(pen=(215, 0, 26), angle=0)

        # Add Items to Viewbox
        self.vb.addItem(self.Image)
        self.vb.addItem(self.xCenter)
        self.vb.addItem(self.yCenter)

        # Plots
        self.PlotY = self.ImageBox.addPlot(row=0, col=1)
        self.PlotX = self.ImageBox.addPlot(row=1, col=0)
        self.PlotX.setXLink(self.Image.getViewBox())
        self.PlotY.setYLink(self.Image.getViewBox())

        # Set Layout
        self.ImageBox.ci.layout.setColumnMaximumWidth(1, 100)
        self.ImageBox.ci.layout.setRowMaximumHeight(1, 100)

    def Main(self):

        self.Plot_CamImage()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def initCam(self, exposureTime):
        self.cam = Basler(namesCamsToUse)
        self.cam.openCommunications()
        self.cam.setCameraParameters(exposureTime)
        self.size = self.cam.getPixelSize(namesCamsToUse[0])
        self.cam.startAquisition()

    def update(self):

        self.updateCamera()
        self.calculateBeamProportion()
        QtGui.QApplication.processEvents()

    def updateCamera(self):

        for i in range(self.average):
            try:
                imgage, nbCam = self.cam.getImage()
                
            except genicam._genicam.LogicalErrorException:
                pass
            if i == 0:
                img = imgage[0].T
            else:
                img = img+imgage[0].T
        img = img/self.average
        self.update_CamImage(img)
        self.calculateCenter(img)
        self.updateImgCalc(img)
        self.cacluateFluence()

    def cacluateFluence(self):
        power = float(self.line_power.text())
        anglePump = float(self.line_angle.text())
        repitionRate = float(self.line_reprate.text())
        FWHMx = imageCalculations["FWHM_X"] * float(self.size[0])
        FWHMy = imageCalculations["FWHM_Y"] * float(self.size[1])
        F =calculateFluenceFromPower(power, anglePump, repitionRate,FWHMx, FWHMy, self.check_chopper.isChecked())
        self.FList.append(F)
        if self.check_average.isChecked():
            if len(self.FList) < int(self.line_nbAverages.text()):
                F = sum(self.FList[:]) / len(self.FList)
            else:
                F = sum(self.FList[-int(self.line_nbAverages.text()):])/int(self.line_nbAverages.text())
        self.label_fluence.setText(str(round(F, 2)))

    def calculateBeamProportion(self):

        try:
            ratio1 = imageCalculations["FWHM_X"] / imageCalculations["FWHM_Y"]
        except ZeroDivisionError:
            ratio1 = 0

        self.label_0ratio.setText(str(round(ratio1, 2)))

        fwhmX = round(imageCalculations["FWHM_X"] * float(self.size[1]), 1)
        self.fwhmXList.append(fwhmX)
        fwhmY = round(imageCalculations["FWHM_Y"] * float(self.size[1]), 1)
        self.fwhmYList.append(fwhmY)
        if self.check_average.isChecked():
            if len(self.fwhmXList) < int(self.line_nbAverages.text()):
                fwhmX = sum(self.fwhmXList[:]) / len(self.fwhmXList)
                fwhmY = sum(self.fwhmYList[:]) / len(self.fwhmYList)
            else:
                fwhmX = sum(self.fwhmXList[-int(self.line_nbAverages.text()):]) / int(self.line_nbAverages.text())
                fwhmY = sum(self.fwhmYList[-int(self.line_nbAverages.text()):]) / int(self.line_nbAverages.text())

        self.label_0fwhmx.setText(str(fwhmX) + ' µm')
        self.label_0fwhmy.setText(str(fwhmY) + ' µm')

        self.label_0fwhmx_slice.setText(str(round((imageCalculations["SliceFWHM_X"] * float(self.size[0])), 1)) + ' µm')
        self.label_0fwhmy_slice.setText(str(round((imageCalculations["SliceFWHM_Y"] * float(self.size[1])), 1)) + ' µm')

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.timer.stop()
            #if self.log:
            #    self.log.closeFile()
            self.Setting.guisave(self.ui, QtCore.QSettings('./GUI/saved.ini', QtCore.QSettings.IniFormat))
            sys.exit()
            event.accept()
        else:
            event.ignore()

    def updateImgCalc(self, img):

        imageCalculations["Image"] = img

        xgauss, ygauss = self.gaussFit()

        if type(xgauss) != int:
            imageCalculations["FWHM_X"] = abs(xgauss[2] * 2.354)
            imageCalculations["Center_GaussFitX"] = xgauss[1]
            self.bFitX = True
        else: 
            self.bFitX = False

        if type(ygauss) != int:
            imageCalculations["Center_GaussFitY"] = ygauss[1]
            imageCalculations["FWHM_Y"] = abs(ygauss[2] * 2.354)
            self.bFitY = True
        else:
            self.bFitY = False

        imageCalculations["SliceY"] = img[int(imageCalculations["Center_GaussFitY"]), :]
        imageCalculations["SliceX"] = img[:, int(imageCalculations["Center_GaussFitX"])]

        xgaussSlice, ygaussSlice = self.gaussFitSlice()
        imageCalculations["SliceFWHM_X"] = abs(xgaussSlice[2] * 2.354)
        imageCalculations["SliceFWHM_Y"] = abs(ygaussSlice[2] * 2.354)

    def calculateCenter(self, img):
        imageCalculations["SumY"] = np.sum(img, axis=0)
        imageCalculations["SumX"] = np.sum(img, axis=1)
        imageCalculations["Center_X"] = len(imageCalculations["SumX"]) / 2
        imageCalculations["Center_Y"] = len(imageCalculations["SumY"]) / 2

    def setPlotBoundaries(self):
        self.PlotY.setYRange(0, len(imageCalculations["SumY"]))
        self.PlotX.setXRange(0, len(imageCalculations["SumX"]))

    def gaussFit(self):

        bFit =True

        ygauss = self.fitGauss(imageCalculations["SumY"], [np.max(imageCalculations["SumY"]), np.argmax(imageCalculations["SumY"]), 0.5, imageCalculations["SumY"][0]])
        xgauss = self.fitGauss(imageCalculations["SumX"], [np.max(imageCalculations["SumX"]), np.argmax(imageCalculations["SumX"]), 0.5,imageCalculations["SumY"][0]])
        QtGui.QApplication.processEvents()
        if type(ygauss) is int or type(xgauss) is int:
            bFit = False

        if bFit:
            imageCalculations["GausYA"] = ygauss[0]
            imageCalculations["GausYB"] = ygauss[1]
            imageCalculations["GausYC"] = ygauss[2]
            imageCalculations["GausXA"] = xgauss[0]
            imageCalculations["GausXB"] = xgauss[1]
            imageCalculations["GausXC"] = xgauss[2]
            imageCalculations["GaussX"] = self.gaus(np.linspace(
                0, len(imageCalculations["SumX"]), len(imageCalculations["SumX"])), xgauss[0], xgauss[1], xgauss[2],xgauss[3])
            imageCalculations["GaussY"] = self.gaus(np.linspace(
                0, len(imageCalculations["SumY"]),
                len(imageCalculations["SumY"])), ygauss[0], ygauss[1], ygauss[2], ygauss[3])
        return xgauss, ygauss

    def gaus(self, x, a, x0, sigma,c):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))+c

    def fitGauss(self, data, init):
        result = False
        i = 0
        while not result and i < 5:
            try:
                i += 1
                popt, pcov = curve_fit(self.gaus, np.linspace(0, len(data),
                                                      len(data)), data, p0=init)
                result = True
            except RuntimeError:
                pass
        if result:
            return popt
        else:
            return 0

    def Plot_CamImage(self):

        colormap = cm.get_cmap(imgColormap)  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        #lut = color
        # Apply the colormap
        self.Image.setLookupTable(lut, update = True)
        self.curve = self.PlotY.plot(pen=(215, 128, 26))
        self.curve2 = self.PlotX.plot(pen=(215, 128, 26))
        self.curve6 = self.PlotX.plot(pen=(255, 0, 0))
        self.curve7 = self.PlotY.plot(pen=(255, 0, 0))
        self.sliceY = self.PlotY.plot(pen=(215, 125, 50))
        self.sliceX = self.PlotX.plot(pen=(215, 125, 50))

        self.XCenter0 = self.PlotX.addLine(x=0, movable=True, pen=(215, 0, 26))
        self.YCenter0 = self.PlotY.addLine(y=0, movable=True, pen=(215, 0, 26))

    def update_CamImage(self, image):

        self.Image.setImage(image, autoLevels = self.check_colorMap.isChecked(), levels = (0, 255))
        self.bar_pixSaturation.setValue(np.max(image))
        try:
            self.curve.setData(x=imageCalculations["SumY"], y=np.arange(len(imageCalculations["SumY"])))
            self.curve2.setData(imageCalculations["SumX"])
            self.curve7.setData(x=imageCalculations["GaussY"], y=np.arange(len(imageCalculations["GaussY"])))
            self.curve6.setData(imageCalculations["GaussX"])

            #self.sliceX.setData(imageCalculations["SliceX"])
            #self.curve7.setData(x=imageCalculations["Slice_GaussY"], y=np.arange(len(imageCalculations["Slice_GaussY"])))
            #self.sliceY.setData(x = imageCalculations["SliceY"],  y=np.arange(len(imageCalculations["GaussY"])))
            #self.curve6.setData(imageCalculations["Slice_GaussX"])

            if self.bZoom:
                self.PlotY.setYRange(self.yMin, self.yMax)
                self.PlotX.setXRange(self.xMin, self.xMax)
                self.init = 0
            elif self.init == 0:
                self.setPlotBoundaries()
                self.init = 1

            self.XCenter0.setValue(imageCalculations["Center_GaussFitX"])
            self.YCenter0.setValue(imageCalculations["Center_GaussFitY"])
            self.xCenter.setValue(imageCalculations["Center_GaussFitX"])
            self.yCenter.setValue(imageCalculations["Center_GaussFitY"])

        except KeyError:
            pass

    def gaussFitSlice(self):

        bFit =True
        ygauss = self.fitGauss(imageCalculations["SliceY"], [20000,750, 100,50])
        
        xgauss = self.fitGauss(imageCalculations["SliceX"], [20000,750, 100,50])
        QtGui.QApplication.processEvents()
        if type(ygauss) is int or type(xgauss) is int:
            bFit = False

        if bFit:
            imageCalculations["Slice_GaussX"] = self.gaus(np.linspace(
                0, len(imageCalculations["SliceX"]), len(imageCalculations["SliceX"])), xgauss[0], xgauss[1], xgauss[2],xgauss[3])
            imageCalculations["Slice_GaussY"] = self.gaus(np.linspace(
                0, len(imageCalculations["SliceY"]),
                len(imageCalculations["SliceY"])), ygauss[0], ygauss[1], ygauss[2], ygauss[3])
        return xgauss, ygauss

def main():

    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
