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


import os, subprocess, pathlib, sys
import tarfile, numpy
from osgeo import gdal


def imread(image):
	img = gdal.Open(image)
	im_array = numpy.array(img.ReadAsArray())
	return numpy.uint16(im_array), img.GetProjection(), img.GetGeoTransform()


def imwrite (fileName, frmt, projection, geotransform, data) :
    drv = gdal.GetDriverByName(frmt)
    rows = data.shape[1]
    cols = data.shape[0]
    out = drv.Create(fileName, rows, cols, 1, gdal.GDT_Float64)
    band = out.GetRasterBand(1)
    band.WriteArray(data)
    band = None
    out.SetProjection(projection)
    out.SetGeoTransform(geotransform)
    out = None  


def extract(path,tilelist):
    tilenamelist = []
    for tile in tilelist:
        tf = tarfile.open(os.path.join(path,tile))
        extraction_path=os.path.join(path,tile[:-7])
        if pathlib.Path(extraction_path).exists()==False:
            pathlib.Path(extraction_path).mkdir(parents=True)
            os.chdir(extraction_path)
            tf.extractall()
        tilenamelist.append(tile[:-7])
    return tilenamelist


def delete_tarfiles(path):
    [tars.append(i for i in os.listdir(path) if i.endswith('gz'))]
    for i in tars:
        os.remove(os.path.join(path,i))
    return 'tar files deleted successfully'


def conversion_decimal(string):
    if string[-1]=='2':
        number = float(string[:-4])*0.01
    elif string[-1]=='3':
        number = float(string[:-4])*0.001
    elif string[-1]=='4':
        number = float(string[:-4])*0.0001
    elif string[-1]=='5':
        number = float(string[:-4])*0.00001
    else:
        print('Error in MTL. Exiting processing')
        sys.exit()
    return number

def parseMTL(path):
    fl = open(path)
    metadata = {}
    for row in fl:
        if "=" in row:
            dt = row.split("=")
            metadata[dt[0].replace(" ","")] = dt[1].replace("\n","")
    return metadata


def atmcorr_landsat(path,tile):
    MLi = 'RADIANCE_MULT_BAND_'
    ALi = 'RADIANCE_ADD_BAND_'
    Mi = 'REFLECTANCE_MULT_BAND_'
    Ai = 'REFLECTANCE_ADD_BAND_'
    SE = 'SUN_ELEVATION'
    K1i = 'K1_CONSTANT_BAND_'
    K2i = 'K2_CONSTANT_BAND_'
    current_folder = os.path.join(path,tile)
    mtlFile = os.path.join(current_folder, tile + '_T1_MTL.txt')
    metaData = parseMTL(mtlFile)

    se = float(metaData[SE])

    for band in [1,2,3,4,5,6,7,8,9]:
        M = Mi+'{0}'.format(band)
        A = Ai+'{0}'.format(band)

        if M not in metaData or A not in metaData:
            continue

        M_val = conversion_decimal(metaData[M])
        A_val = float(metaData[A])

        image = [i for i in os.listdir(current_folder) if i.endswith('B{0}.TIF'.format(band))]
        img = imread(os.path.join(current_folder,image[0]))

        spectral_reflectance_band = M_val*img[0]+A_val
        toa_reflectance_band = spectral_reflectance_band/numpy.sin(se * numpy.pi/180.)
        toa_reflectance_band = numpy.where(img[0]==0,0,toa_reflectance_band)
        filename = os.path.join(current_folder,tile+'_B{0}_refl.TIF'.format(band))
        imwrite (filename, 'GTiff', img[1], img[2], toa_reflectance_band)
        
    for band in [10, 11]:
        ML = MLi+'{0}'.format(band)
        AL = ALi+'{0}'.format(band)
        K1 = K1i+'{0}'.format(band)
        K2 = K2i+'{0}'.format(band)

        if ML not in metaData or AL not in metaData or K1 not in metaData or K2 not in metaData:
            continue

        ML_val = conversion_decimal(metaData[ML])
        AL_val = float(metaData[AL])
        K1_val = float(metaData[K1])
        K2_val = float(metaData[K2])

        image = [i for i in os.listdir(current_folder) if i.endswith('B{0}.TIF'.format(band))]
        img = imread(os.path.join(current_folder,image[0]))
        
        spectral_radiance_band = ML_val*img[0]+AL_val  
        toa_brightness_temperature = K2_val/numpy.log((K1_val/spectral_radiance_band)+1)-273.
        toa_brightness_temperature = numpy.where(img[0]==0,0,toa_brightness_temperature)
        filename = os.path.join(current_folder,tile+'_B{0}_temp.TIF'.format(band))
        imwrite (filename, 'GTiff', img[1], img[2], toa_brightness_temperature)

#=============================================
PATH = "/path/to/imagery/folder/"
years = [ '2013','2014', '2015', '2016','2017','2018', '2019', '2020', '2021','2022' ]

for year in years:
    path_process=os.path.join(PATH,year)
    print(path_process)

    tarBalls = [f for f in os.listdir(path_process) if f.endswith(".tar")]
    
    subfolders = extract(path_process,tarBalls)
    
    for scene in subfolders:
        atmcorr_landsat(path_process,scene)
