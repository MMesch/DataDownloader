#!/usr/bin/env python
#This file is part of DataDownloader -> see the LICENSE file
import obspy.core as co
from obspy.fdsn.client import Client
import argparse
import os

#==== MAIN FUNCTION ====
def main():
    """
    This script downloads waveforms, events and station information
    for a given time period and given station search string.
    More info from:

    command: ./downloader.py --help
    """
    args      = readArguments()
    baseDir   = createDir(args)
    download(args,baseDir)

#==== download function ====
def download(**kwargs):
    """
    downloads event, station and waveform data for the given time range and
    station id

    args:
    starttime = UTCDateTime()
    endtime   = UTCDateTime()
    network   = '*'
    station   = 'ANMO,DBSD'
    location  = '*'
    channel   = 'VHZ'
    minradius = minimum station distance 
    maxradius = maximum station distance
    """
    events    = getEvents(kwargs)
    inventory = getStationXML(kwargs,events)
    waveforms = getWaveforms(kwargs, inventory)
    finalOutput(kwargs['baseDir'], waveforms, inventory, events)

#==== READING AND DIRECTORIES ====
def readArguments():
    parser = argparse.ArgumentParser(description='downloads data from IRIS using obspy.')
    parser.add_argument('starttime',   help='start time (e.g.: 2005-10-7-12-5)')
    parser.add_argument('endtime',     help='end time (e.g.: 2005-11-0-0-0)')
    parser.add_argument('identity',    help='network.station.location.channel (e.g. IU.AFI.00.VHZ)')
    parser.add_argument("--minradius", help="minimum distance", default=None)
    parser.add_argument("--maxradius", help="maximum distance", default=None)
    args = parser.parse_args()
    network, station, location, channel = args.identity.split('.')

    argdict = {}
    argdict['starttime']  = co.UTCDateTime(args.starttime)
    argdict['endtime']    = co.UTCDateTime(args.endtime)
    argdict['network']    = network
    argdict['station']    = station
    argdict['location']   = location
    argdict['channel']    = channel
    argdict['minradius']  = args.minradius
    argdict['maxradius']  = args.maxradius
    return argdict

#---- create Directories ----
def createDir(args):
    dirName = os.path.join('data','%s-%s'%(args['starttime'].date,args['endtime'].date))

    try:
      os.makedirs(dirName)
      print 'saving under path:',dirName
      return dirName
    except OSError, err:
      if err.errno == 17:
          print '\n directory exists!'
          return dirName
      else:
          print err
          print '\nCan\'t create directory tree'
          raise

#==== DOWNLOAD FROM IRIS ====
def getStationXML(args,events):
    print '\n---- checking data availability ----'
    print args
    client = Client('Iris')
    if args['minradius'] or args['maxradius']:
        lat = events[0].origins[0].latitude
        lon = events[0].origins[0].longitude
        print 'searching around event location:',lat,lon
        inventory = client.get_stations(level='response',latitude=lat,longitude=lon,**args)
    else:
        inventory = client.get_stations(level='response',**args)
    print '\nData is available for: '
    print inventory
    return inventory

def getEvents(args):
    client = Client('Iris')
    events = client.get_events(minmag=5.0,starttime=args['starttime'],endtime=args['endtime'])
    print 'found these events:'
    print events
    return events

def getWaveforms(args, inventory):
    traces = inventory.get_contents()['channels']
    ntraces = len(traces)
    print 'Downloading {:d} waveforms'.format(ntraces)
    print traces

    t1 = args['starttime']
    t2 = args['endtime']
    downloadlist = [tuple(trace.split('.'))+(t1,t2) for trace in traces]

    parameters = {'attach_response':False,
                  #'minimumlength':10.*3600.,
                  'longestonly':True}
    client = Client('Iris')
    waveforms = client.get_waveforms_bulk(downloadlist,**parameters)
    print 'downloaded waveforms'
    print waveforms
    return waveforms

#==== FINAL OUTPUT ====
def finalOutput(baseDir, waveforms, inventory, events):
    print '\n==== DOWNLOAD DONE. ====\n\n'
    waveforms.write(baseDir+'/waveforms.mseed',format='MSEED')
    inventory.write(baseDir+'/stations.xml',format='STATIONXML')
    events.write(baseDir+'/events.xml',format='QUAKEML')
    print 'waveforms, stationxml and quakeml files saved in dir:',baseDir

#==== EXECUTE ====
if __name__== "__main__":
    main()
