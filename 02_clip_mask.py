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


import os, sys


#=============================================
PATH = '/path/to/imagery/folder/'
years = ['2013', '2014','2015','2016','2017','2018','2019', '2020', '2021', '2022']
aoi_path = '/path/to/AOI/extent/shapefile/'

for year in years:
    path_process=os.path.join(PATH,year)
    print(path_process)
    
    subfolders = [f for f in os.listdir(path_process) if not f.endswith(".tar")]
    
    for scene in subfolders:
        imagelist=[]
        current_path1 = os.path.join(path_process,scene)
        print(current_path1)
        for i in os.listdir(current_path1):
            if i.endswith('refl.TIF') or i.endswith('temp.TIF'):
                imagelist.append(i)
        for image in imagelist:
            current_path2 = os.path.join(current_path1,image)
            os.system('gdalwarp -srcnodata 0 -overwrite -crop_to_cutline -cutline {0} {1} {2}'.format(aoi_path,current_path2,current_path2[:-4]+'_clip.tif'))

