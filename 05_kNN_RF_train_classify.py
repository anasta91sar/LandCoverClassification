"""
   Copyright (C) 2023  Anastasia Sarelli an.sarelli at gmail dot com
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import datetime
from multiprocessing import Process, Manager
from osgeo import gdal


def imread(image):
	img = gdal.Open(image)
	im_array = np.array(img.ReadAsArray())
	return im_array, img.GetProjection(), img.GetGeoTransform() 

def imwrite (fileName, frmt, projection, geotransform, data) :
    drv = gdal.GetDriverByName(frmt)
    rows = data.shape[1]
    cols = data.shape[0]
    out = drv.Create(fileName, rows, cols, 1, gdal.GDT_Float32)
    band = out.GetRasterBand(1)
    band.WriteArray(data)
    band = None
    out.SetProjection(projection)
    out.SetGeoTransform(geotransform)
    out = None  

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


class ComputeModels():
    
    def __del__(self):
        self.log.close()
        self.log = None
        
    def __init__(self, dataPath, outPath, samplesFile, cols2use, mode, seasonDivision=0):

        self.PATH_in = dataPath
        self.classification_output_path = outPath
        os.makedirs(self.classification_output_path, exist_ok=True)

        self.classes = {"artificial":0,"bare_soil":1,"cropland":2,"dense_forest":3, "grassland":4,"low_density_urban":5,
                        "low_sparse_vegetation":6,"water":7}
        self.classNames = ["artificial", "bare_soil", "cropland", "dense_forest", "grassland", "low_density_urban",
                           "low_sparse_vegetation", "water"]
        self.samplesFile = samplesFile
        self.cols2use = cols2use
        self.mode = mode
        logFile = 'log_'+'_' + str(mode) +'.txt'
        
        self.log = open(os.path.join(self.classification_output_path, logFile),"w+") 

        
        samples = open(self.samplesFile, "r")
        self.dataset = read_csv(samples,low_memory=False)
        seasonData = {'all': [[], []]}
        if seasonDivision == 1:
            seasonData = {'all':[[],[]],'summer':[[],[]],'autumn':[[],[]],'winter':[[],[]],'spring':[[],[]]}
        tmpVals = self.dataset.values
        selectedVals = self.dataset[cols2use].values
        tmpLabels = [x.replace(" ","") for x in tmpVals[:,-1]]
        if seasonDivision == 1:
            for j in range (0, tmpVals.shape[0]):
                                
                key = None
                if tmpVals[j][1] == 12 or tmpVals[j][1] == 1 or tmpVals[j][1] == 2:
                    key = "winter"
                elif tmpVals[j][1] == 3 or tmpVals[j][1] == 4 or tmpVals[j][1] == 5:
                    key = "spring"
                elif tmpVals[j][1] == 6 or tmpVals[j][1] == 7 or tmpVals[j][1] == 8:
                    key = "summer"
                elif tmpVals[j][1] == 9 or tmpVals[j][1] == 10 or tmpVals[j][1] == 11:
                    key = "autumn"
                    
                seasonData[key][0].append(selectedVals[j])
                seasonData[key][1].append(self.classes[tmpLabels[j] ])

        seasonData["all"][0] = selectedVals
        seasonData["all"][1] = [self.classes[x] for x in tmpLabels]

        # Split-out validation dataset
        self.trainXMin = {}
        self.trainXMax = {}
        self.X_train = {}
        self.X_validation = {}
        self.Y_train = {}
        self.Y_validation = {}
        for season in seasonData:
            seasonData[season][0] = np.array(seasonData[season][0])
            seasonData[season][1] = np.array(seasonData[season][1]).reshape(-1,1)
            self.trainXMin[season] = seasonData[season][0].min(axis = 0)
            self.trainXMax[season] = seasonData[season][0].max(axis = 0)
        
            self.X_train[season], self.X_validation[season], self.Y_train[season], self.Y_validation[season] = train_test_split(seasonData[season][0], seasonData[season][1], test_size=0.20, random_state=1, shuffle=True)
            self.Y_train[season] = self.Y_train[season].flatten()
            self.Y_validation[season] = self.Y_validation[season].flatten()

    
    def trainKNeighbors(self):
        self.KNmodel = {}
        for season in self.X_train:
            self.KNmodel[season] = KNeighborsClassifier()
            self.KNmodel[season].fit(self.X_train[season], self.Y_train[season])

    def trainRandomForest(self):
        self.RandomForestModel = {}

        for season in self.X_train:
            self.RandomForestModel[season] = RandomForestClassifier()
            self.RandomForestModel[season].fit(self.X_train[season], self.Y_train[season])
            
        
    def predictModel(self, model, xval, yval):

        for season in xval:
            self.log.write('\n\n{} prediction results for season: {}'.format(model[season],season) + '\n')
            model[season].n_jobs = 24
            predictions = model[season].predict(xval[season])
            self.log.write(str(accuracy_score(yval[season], predictions)) + '\n')
            self.log.write(str(confusion_matrix(yval[season], predictions)) + '\n')
            self.log.write(str(classification_report(yval[season], predictions)) + '\n')


            showClasses = [self.classNames[i].replace("_"," ") for i in model[season].classes_]



            cm = confusion_matrix(yval[season], predictions)
            cm = cm/ cm.sum(axis=1)
            cm = np.round(cm,3)
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.6)
            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Reference', fontsize=18)
            plt.title('Confusion Matrix for season: {0}'.format(season), fontsize=23)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

            ax.set_yticks(list(range(len(showClasses))), showClasses, fontsize=15, rotation=30)
            ax.set_xticks(list(range(len(showClasses))), showClasses, fontsize=15, rotation=20)
            plt.subplots_adjust(top=0.88)

            plt.savefig(os.path.join(self.classification_output_path,'confusion_matrix_{0}.jpg'.format(season)))
            plt.close()

        
        
    def writeClassificationResult(self, model, modelName, demPath=None, slopePath=None, tpiPath=None):

        years = ['2018',] #'2014','2015','2016','2017','2018','2019'
        for year in years:
            path_process=os.path.join(self.PATH_in,year)
            print(path_process)
    
            subfolders = [f for f in os.listdir(path_process) if not f.endswith(".tar")]
    
            for scene in subfolders:
                inData = os.path.join(path_process,scene)
                pathRow = scene.split("_")[2]
                inListFiles = os.listdir(inData)
                inListArray = []
                projection = None
                geoTransform = None
                for band in self.cols2use:
                    for fileName in inListFiles:
                        if band.upper() in fileName and fileName.endswith(".tif") and "clip" in fileName:
                            tmpDataset = gdal.Open(os.path.join(inData, fileName))
                            inListArray.append(tmpDataset.ReadAsArray())
                            projection = tmpDataset.GetProjection()
                            geoTransform = tmpDataset.GetGeoTransform()
                
                if "elevation" in self.cols2use:
                    demImage, demProjection, demGeoTransform = imread(demPath)
                    inListArray.append(demImage)
                    
                    
                if "slope" in self.cols2use:
                    slopeImage, slopeProjection, slopeGeoTransform = imread(slopePath)
                    inListArray.append(slopeImage)
                    
                if "tpi" in self.cols2use:
                    tpiImage = imread(os.path.join(tpiPath))[0]
                    inListArray.append(tpiImage)
                    
                inDataset = np.array(inListArray)
                normalizedDataset = inDataset.T.reshape(inDataset.shape[1]*inDataset.shape[2], inDataset.shape[0])
                model["all"].n_jobs = 8

                outPath = os.path.join(self.classification_output_path,year)
                os.makedirs(outPath, exist_ok=True)

                outBand = model["all"].predict(normalizedDataset)

                imwrite(os.path.join(outPath,scene +'_'+'_'.join(self.cols2use)+'_{0}_class.tif'.format(modelName)), "GTiff", projection, geoTransform, outBand.reshape((inDataset.shape[2], inDataset.shape[1])).T)
                print("ok!")
                return

 
def writeClassificationResultTimeseries(self, model, modelName, dates, uniqueBands, demPath=None, slopePath=None, tpiPath=None ):
        spectralBands = uniqueBands
        if "elevation" in uniqueBands:
            spectralBands = uniqueBands[0:-3]
        bandFiles = []

        for band in spectralBands:
            for date in dates:
                mergeDate = str(date[0])+date[1].replace(" ","")+date[2].replace(" ","")
                dataPath = os.path.join(self.PATH_in, str(date[0]))
                dataset = [f for f in os.listdir(dataPath) if not f.endswith(".tar") and mergeDate == f.split("_")[3]][0]
                dataPath = os.path.join(dataPath, dataset)
                file = [f for f in os.listdir(dataPath) if "clip" in f and band.upper() in f and ".xml" not in f][0]
                bandFiles.append(os.path.join(dataPath, file))
                #bandFiles.append(file)

        #appending elevation data
        if "elevation" in uniqueBands:
            bandFiles.append(demPath)
            bandFiles.append(slopePath)
            bandFiles.append(tpiPath)

        #reading reference image
        #tmpDt = gdal.Open(bandFiles[0])
        sampleFile = gdal.Open(bandFiles[0])
        dims = [sampleFile.RasterXSize,sampleFile.RasterYSize]

        drv = gdal.GetDriverByName("GTiff")
        outFile = os.path.join(self.classification_output_path, "output.tif")
        outDataset = drv.Create(outFile, dims[0], dims[1], 1, gdal.GDT_Float32)
               
  gt = list(sampleFile.GetGeoTransform())

        outDataset.SetGeoTransform(gt)
        outDataset.SetProjection(sampleFile.GetProjection())

        outDataset = None
        sampleFile = None
        nThreads = 8
        chunks = chunkIt(range(dims[1]), nThreads)
        threads = list(range(nThreads))

        for thread in range(nThreads):
            print(chunks[thread])
            threads[thread] = Process(target=processRegion, args=(bandFiles, chunks[thread], dims, model, outFile))
            threads[thread].start()

        for trd in threads:
            trd.join()

        return 0


    
def main():
    filePath = '/path/to/imagery/folder/'
    outPath = "/output/path/of/results/"
    dataset = "/path/to/samples/dataset_cloud_free.csv"
    demPath = '/path/to/dem_aligned.tif'
    slopePath = '/path/to/slope_aligned.tif'
    tpiPath = '/path/to/tpi_aligned.tif'

    trainingModes = {

        "multispectral": {
            "columns_to_use": ['b2', 'b3', 'b4', 'b5', 'b6', 'b7']
        },
        "multispectral_thermal": {
            "columns_to_use": ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', "b10", "b11"]
        },
        "multispectral_terrain": {
            "columns_to_use": ['b2', 'b3', 'b4', 'b5', 'b6', 'b7','elevation','slope','tpi']
        },
        "multispectral_thermal_terrain": {
            "columns_to_use":['b2', 'b3', 'b4', 'b5', 'b6', 'b7',"b10", "b11",'elevation','slope','tpi']
        }
    }
    for algorithm in ["RF", "kNN", ]:
        print("Algorithm: ", algorithm)
        for mode in trainingModes:
            print("Performing mode: ", mode)
            a = ComputeModels(filePath, os.path.join(outPath,*[mode, algorithm]) , dataset,
                                                     trainingModes[mode]["columns_to_use"], mode, 0)
            model=None
            if(algorithm == "kNN"):
                a.trainKNeighbors()
                model = a.KNmodel
            elif(algorithm == "RF"):
                a.trainRandomForest()
                model = a.RandomForestModel

            a.predictModel(model, a.X_validation, a.Y_validation)
            a.writeClassificationResult(model,algorithm, demPath, slopePath, tpiPath)
            #computing classification results


if __name__ == "__main__":
    main()