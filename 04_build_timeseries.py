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


import psycopg2, numpy as np

cnStr = "dbname=thesis user=postgres"
cn = psycopg2.connect(cnStr)


query = "SELECT DISTINCT year, month, day FROM dataset_cloud_free ORDER BY year, month,day"
cursor = cn.cursor()
cursor.execute(query)
dates = cursor.fetchall()
dateCount = len(dates)
print(dateCount)

query = "SELECT DISTINCT  elevation, slope, tpi, poly_id, pixel_id, class FROM cloudfree ORDER BY poly_id, pixel_id"
cursor = cn.cursor()
cursor.execute(query)
polyIDs = cursor.fetchall()
cols = ["b2", "b3", "b4", "b5", "b6", "b7", "b10", "b11"]
outFile = open("/dataset_cloud_free_timeseries.csv", "w")

header = []
for col in cols:
        for date in dates:
            header += [col+"({0}-{1}-{2})".format(*date)]

header+=["elevation", "slope", "tpi", "poly_id", "pixel_id", "class"]
outFile.write(",".join(header))
outFile.write("\n")


for rowDt in polyIDs:
    print(rowDt[-3], rowDt[-2])
    #reading multispectral info
    query = """SELECT {0}
        FROM cloudfree dv 
        WHERE poly_id='{1}' and pixel_id = '{2}'
        ORDER BY YEAR,MONTH,day""".format(",".join(cols), rowDt[-3], rowDt[-2])
    cursor = cn.cursor()
    cursor.execute(query)
    timeseries = cursor.fetchall()
    timeseries = np.array(timeseries).T.flatten()

    #appending terrain, class, and id info

    row = timeseries.tolist() + list(rowDt)

    outFile.write(",".join(row))
    outFile.write("\n")
outFile.close()







