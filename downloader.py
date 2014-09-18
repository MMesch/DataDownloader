#!/usr/bin/env python
#This file is part of DataDownloader -> see the LICENSE file
import obspy.core as co
import numpy as np
from obspy.iris import Client as Iris
import argparse
import os
import glob

#==== MAIN FUNCTION ====
def main():
   """
   this script does:
   I. first download continuous data from IRIS
   II. save DATA as SAC
   III. download PAZ files
   Errors during the download process are saved in the boolean array "mask"
   This mask could be used to check whether files already exist
   command: ./downloader.py --help
   """
   args = readArguments()
   dataList,mask = checkAvailability(args)
   baseDir, dataList, mask = createDir(args, dataList, mask) #create directory and mask already existing stations
   mask = getWaveforms(baseDir,dataList,mask)
   finalOutput(dataList,mask)
   return 0

#==== READING AND DIRECTORIES ====
def readArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('mintime', help='start time (e.g.: 2005-10-7-12-5)')
    parser.add_argument('maxtime', help='start time (e.g.: 2005-11-0-0-0)')
    parser.add_argument('identity', help='network.station.location.channel (e.g. IU.AFI.00.VHZ)')
    args = parser.parse_args()
    mintime = co.UTCDateTime(args.mintime)
    print mintime
    maxtime = co.UTCDateTime(args.maxtime)
    network, station, location, channel = args.identity.split('.')
    return network, station, location, channel, mintime, maxtime

#---- create Directories ----
def createDir(args, dataList, mask):
    mintime = args[4]
    maxtime = args[5]
    dirName = 'data/%s-%s'%(mintime.date,maxtime.date)
    wfdirectory = dirName + '/Waveforms'
    pazdirectory = dirName + '/PAZ'

    if os.path.isdir(wfdirectory) and os.path.isdir(pazdirectory):
        print 'directory exists. Checking for already existing files...'
        saclist = [f.rsplit('/')[-1] for f in glob.glob(wfdirectory + '/*.sac')]
        pazlist = [f.rsplit('/')[-1] for f in glob.glob(pazdirectory + '/*.paz')]
        for i,stat in enumerate(dataList):
            sacname = '.'.join(stat[:4])+'.sac'
            pazname = '.'.join(stat[:4])+'.paz'
            if (sacname in saclist) and (pazname in pazlist):
                mask[i] = False
        print '%d of %d stations already existed.'%(np.count_nonzero(np.invert(mask)),len(dataList))
        newDataList = [stat for i,stat in enumerate(dataList) if mask[i] == True]
        return dirName, newDataList , np.ones(len(newDataList),dtype=np.bool)

    else:
        try:
          os.makedirs(dirName)
          os.makedirs(dirName + '/Waveforms/')
          os.makedirs(dirName + '/PAZ/')
          return dirName, dataList
        except Exception:
          print 'Can\'t create directory tree'
          return dirName, dataList, np.ones(len(dataList),dtype=np.bool)
#---- check for existing waveforms or PAZ files ----

#==== DOWNLOAD FROM IRIS ====
#---- check availability ----
def checkAvailability(args):
    print 'checking data availability for net: %s stat:%s loc: %s chan: %s\n t1 = %s, t2 = %s:'%args
    try:
      client_iris = Iris()
      availData = client_iris.availability(*args)
      print 'Data is available for: '
      print availData
      dataList = [tuple(stat.split()) for stat in availData.strip().split('\n')]
      mask = np.ones(len(dataList),dtype=bool)
      return dataList, mask
    except Exception:
      print 'Exception in checking availability of waveforms'; raise
      return 1

#---- downloading waveforms ----
def getWaveforms(baseDir, dataList, mask):
    NWaveforms = len(dataList)
    wfdirectory = baseDir + '/Waveforms'
    pazdirectory = baseDir + '/PAZ'
    client_iris = Iris()
    for i,data in enumerate(dataList):
      #---- download waveform ----
      if mask[i]:
        try:
          waveform = client_iris.getWaveform(*data)[0]
        except Exception:
          print 'NO Waveform AVAILABLE %d/%d'%(i+1,NWaveforms) + ' for: %s.%s.%s.%s'%data[:-2]
          mask[i] = False
      #---- download paz and write to File ----
      if mask[i]:
        try:
          paz = client_iris.sacpz(*data)
          pazfile = open(pazdirectory + '/%s.%s.%s.%s.paz'%data[:-2],'w')
          pazfile.write(paz)
          pazfile.close()
          print 'Saving Paz file %d/%d'%(i+1,NWaveforms) + ' for: %s.%s.%s.%s  ---> DONE'%data[:-2]
          waveform.write(wfdirectory + '/%s.%s.%s.%s.sac'%data[:-2],'SAC')
          print 'Saving Waveform %d/%d'%(i+1,NWaveforms) + ' for: %s.%s.%s.%s  ---> DONE'%data[:-2]
        except Exception,e:
          print e
          print 'NO Paz AVAILABLE %d/%d'%(i+1,NWaveforms) + ' for: %s.%s.%s.%s  ---> DONE'%data[:-2]
          mask[i] = False
    return mask

#==== FINAL OUTPUT ====
def finalOutput(dataList,mask):
    print '\n==== DOWNLOADING DONE. ====\n\n'
    print 'Some Statistics:'
    print '%d of %d stations were downloaded.'%(np.count_nonzero(mask),len(dataList))
    print 'failed or existing stations are:'
    ifailed = np.nonzero(np.invert(mask))[0]
    for i in ifailed:
        print dataList[i],' '
    print ''

if __name__== "__main__":
    main()
