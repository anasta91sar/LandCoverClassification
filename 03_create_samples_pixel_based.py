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

import os, math, numpy

import numpy as np
from osgeo import ogr, gdal, osr
from AlignToGrid import AlignToGrid

def geomRasterizer(id, geom, refGrid, resDict, resolution, parcEPSG=32634, destEPSG=32634):
    drv = ogr.GetDriverByName('MEMORY')
    ds = drv.CreateDataSource("tmp")
    parcOSR = osr.SpatialReference()
    parcOSR.ImportFromEPSG(int(parcEPSG))

    destOSR = osr.SpatialReference()
    destOSR.ImportFromEPSG(int(destEPSG))

    lr = ds.CreateLayer("tmpftlr",parcOSR)
    idField = ogr.FieldDefn("id", ogr.OFTInteger64)
    lr.CreateField(idField)

    #create new ogr feature
    parc = ogr.Feature(lr.GetLayerDefn())
    parc.SetField("id",id)
    parc.SetGeometry(geom)
    allignedGrid = AlignToGrid(parc, refGrid)
    grd = allignedGrid.process(vector=True)
    drv = gdal.GetDriverByName("MEM")

    tmpDataset =  drv.Create("__del\\{0}.tif".format(parc.GetField("id")),
    int((grd[1][0] - grd[0][0]) / resolution), int((grd[0][1] - grd[1][1]) / resolution), 1, gdal.GDT_Byte)
    tmpDataset.SetProjection(parc.GetGeometryRef().GetSpatialReference().ExportToWkt())
    tmpDataset.SetGeoTransform((grd[0][0], resolution, 0, grd[0][1], 0, -resolution))

    # create temporary dataset
    ogrDrv = ogr.GetDriverByName("MEMORY")
    memVSource = ogrDrv.CreateDataSource(str(parc.GetField("id")))
    memVLayer = memVSource.CreateLayer("tmp", destOSR, geom_type=ogr.wkbPolygon)
    tmpFtDefn = memVLayer.GetLayerDefn()
    ft = ogr.Feature(tmpFtDefn)
    ft.SetGeometry(parc.geometry())
    memVLayer.CreateFeature(ft)
    gdal.RasterizeLayer(tmpDataset, [1], memVLayer, burn_values=[1, ])
    resDict[parc.GetField("id")] = {"gt":tmpDataset.GetGeoTransform(), "prj":tmpDataset.GetProjection(),
        "mask":tmpDataset.ReadAsArray(), "RasterXSize": tmpDataset.RasterXSize,
        "RasterYSize":tmpDataset.RasterYSize}
    tmpDataset = None

def imBlockRead(path, res, id_):
    tmpDt = gdal.Open(path)
    tmpGt = tmpDt.GetGeoTransform()
    col, row = xyToRowCol(res[id_]["gt"][0], res[id_]["gt"][3], tmpGt)
    tmpArray = tmpDt.GetRasterBand(1).ReadAsArray(col, row, res[id_]["RasterXSize"], res[id_]["RasterYSize"])
    if tmpArray is None:
        return None

    tmpArray = tmpArray.astype(float)
    tmpArray[res[id_]["mask"] == 0] = numpy.nan
    tmpArray[tmpArray == tmpDt.GetRasterBand(1).GetNoDataValue()] = numpy.nan
    return tmpArray



def imread(image):
	img = gdal.Open(image)
	im_array = numpy.array(img.ReadAsArray())
	return im_array, img.GetProjection(), img.GetGeoTransform()

def xyToRowCol(X, Y, gt):
    y = int((Y - gt[3]-gt[4]/gt[1]*X+gt[0]*gt[4]/gt[1])/(gt[5]-(gt[2]*gt[4]/gt[1])))
    x = int((X-gt[0]-gt[2]*y)/gt[1])
    return [x,y]


    #=============================================
aoi_path = '/path/to/AOI/extents/'
reference_image = "/path/to/reference/image/LC09_L1TP_185033_20220531_20220601_02/LC09_L1TP_185033_20220531_20220601_02_B6_refl_clip.tif"

PATH = '/path/to/imagery/folder/'
DEM_PATH = '/path/to/dem_aligned.tif'
SLOPE_PATH = '/path/to/slope_aligned.tif'
TPI_PATH = '/path/to/tpi_aligned.tif'

aoi_path = '/polygon/samples/training_poly.gpkg'
shp_name = 'training_poly_utm'

samples_path = '/folder/for/the/pixelbased/samples/'

TEMP_fold = '/temporary/folder/'
im_path_out = os.path.join(TEMP_fold,'output.tif')
sample_path_out = os.path.join(TEMP_fold,'sample.shp')
dem_path_out = os.path.join(TEMP_fold,'dem.tif')
slope_path_out = os.path.join(TEMP_fold,'slope.tif')
tpi_path_out = os.path.join(TEMP_fold,'tpi.tif')
    #=============================================
years = ['2013', '2014','2015','2016','2017','2018','2019', '2020', '2021', '2022']
names = ['year','month','day','poly_id', 'pixel_id', 'B2','B3','B4','B5','B6','B7','B10','B11','elevation','slope','tpi','class']

classes = ["artificial","bare_soil","cropland","dense_forest","low_density_urban","low_sparse_vegetation","water"]

    #=============================================
txt_file = open(os.path.join(samples_path,'dataset.csv'),"w+")
txt_file.write(",".join(names))
txt_file.write("\n")
file = ogr.Open(aoi_path)
shape = file.GetLayer()
refImage = gdal.Open(reference_image)
gt = refImage.GetGeoTransform()
refImage = None

for feature in shape:
    cat = feature.GetField("class")
    id_ = feature.GetFID()
    print("Processing id: ", id_)
    res = {}
    geomRasterizer(id_, feature.geometry(), reference_image, res, gt[1])

    rawDt = [None]*11

    rawDt[-3] = imBlockRead(DEM_PATH, res, id_).flatten()
    rawDt[-2] = imBlockRead(SLOPE_PATH, res, id_).flatten()
    rawDt[-1] = imBlockRead(TPI_PATH, res, id_).flatten()


                        
    for year in years:
        path_process=os.path.join(PATH,year)
        subfolders = [f for f in os.listdir(path_process) if not f.endswith(".tar")]

        for scene in subfolders:
            current_path1 = os.path.join(path_process,scene)

            date = os.path.split(current_path1)[1].split("_")[3]

            bandId = 0
            for attr in names_test[4::]:
                for i in os.listdir(current_path1):
                    if i.endswith('.tif') and attr in i:
                        path = os.path.join(current_path1, i)
                        rawDt[bandId] = imBlockRead(path, res, id_).flatten()
                        bandId += 1

            rowOffset = 5
            pixelCount = rawDt[bandId].shape[0]

            for i in range(pixelCount):
                if np.isnan(rawDt[0][i]):
                    continue

                new_row = list(range(len(names_test)))
                new_row[0] = date[0:4]
                new_row[1] = date[4:6]
                new_row[2] = date[6:8]
                new_row[3] = str(id_)
                new_row[4] = str(i)
                isNone = False


                k = 0
                for bnd in rawDt:
                    new_row[rowOffset+k] = str(bnd[i])
                    k += 1

                new_row[16] = cat
                txt_file.write(', '.join(new_row) + '\n')

txt_file.close()