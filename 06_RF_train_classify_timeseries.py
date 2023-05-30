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


import psycopg2, os
from train_classify_v2 import ComputeModels


def main():
    filePath = '/path/to/imagery/folder/'
    dataset = "/path/to/samples/dataset_cloud_free_timeseries.csv"
    outPath = "/output/path/for/results"
    demPath = '/path/to/dem_aligned.tif'
    slopePath = '/path/to/slope_aligned.tif'
    tpiPath = '/path/to/tpi_aligned.tif'

    cnStr = "dbname=thesis user=postgres"
    cn = psycopg2.connect(cnStr)
    query = "SELECT DISTINCT '('||year || '-' || month || '-' || day || ')', year, month, day FROM dataset_cloud_free ORDER BY year, month, day"
    cursor = cn.cursor()
    cursor.execute(query)
    dates = cursor.fetchall()

    terrainCols = ['elevation','slope','tpi']

    trainingModes = {
        "multispectral": {
            "columns_to_use": ['b2', 'b3', 'b4', 'b5', 'b6', 'b7']
        },
        "multispectral_thermal": {
            "columns_to_use": ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', "b10", "b11"]
        },

        "multispectral_terrain": {
            "columns_to_use": ['b2', 'b3', 'b4', 'b5', 'b6', 'b7']+terrainCols
        },
        "multispectral_thermal_terrain": {
            "columns_to_use":['b2', 'b3', 'b4', 'b5', 'b6', 'b7',"b10", "b11"]+terrainCols
        }
    }


    for mode in trainingModes:
        requestCols = []
        parseDate = []
        appendDate = True
        for col in trainingModes[mode]["columns_to_use"]:
            if col not in terrainCols:
                for row in dates:
                    requestCols.append(col+row[0])
                    if(appendDate):
                        parseDate.append(row[1:4])
                appendDate = False

        for col in terrainCols:
            if col in trainingModes[mode]["columns_to_use"]:
                requestCols.append(col)

  
        algorithm = "RF"
        a = ComputeModels(filePath, os.path.join(outPath, *[mode, algorithm]), dataset, requestCols, mode, 0)
        a.trainRandomForest()
        model = a.RandomForestModel
        a.predictModel(model, a.X_validation, a.Y_validation)
        a.writeClassificationResultTimeseries(model,algorithm, parseDate, trainingModes[mode]["columns_to_use"], demPath, slopePath, tpiPath)
  
    return 0


if __name__ == "__main__":
    main()

