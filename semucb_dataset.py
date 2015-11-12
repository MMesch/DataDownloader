#!/usr/bin/env python
"""
this script redownloads waveform data from iris, etc to get an idea about the pre-event
noise level of each data trace from SEMUCB
"""
from obspy.fdsn.client import Client
from ndk_rec_file import NDKFile,RecFile
import os

#==== MAIN FUNCTION ====
def main():
    #==== INITIALIZE ====
    #---- parameters ----
    redownload = False
    basedir    = 'semucb_dataset'
    datadir    = os.path.join(basedir,'data')
    t1,t2      = -3600.,10000.

    #---- input files ----
    fname_evcatalogue = os.path.join(basedir,'evcatalogue_semucb.ndk')
    fname_stcatalogue = os.path.join(basedir,'receivers.dat')

    #==== PREPARE DOWNLOAD LIST ====
    print 'reading station list...'
    stcatalogue = RecFile(fname_stcatalogue)
    print 'retrieved {:d} stations'.format(stcatalogue.ndata)
    unique_stations = set(stcatalogue.stations)
    print 'unique stations: {}'.format(len(unique_stations))

    print 'reading event catalogue...'
    evcatalogue = NDKFile(fname_evcatalogue)
    print 'found {:d} events'.format(evcatalogue.nevents)

    create_dir(datadir)

    #==== LOOP THROUGH EVENTS ====
    client = Client('Iris')
    for name,event in zip(evcatalogue.names,evcatalogue.events):
        print '---- {:s} ----'.format(name)
        fname_events   = '{:s}_events.xml'.format(name)
        fname_stations = '{:s}_stations.xml'.format(name)
        fname_mseed    = '{:s}_waveforms.mseed'.format(name)

        path_stations = os.path.join(datadir,fname_stations)
        path_events   = os.path.join(datadir,fname_events)
        path_mseed    = os.path.join(datadir,fname_mseed)

        #---- check if data exist ----
        if os.path.exists(path_mseed)\
        and os.path.exists(path_events)\
        and os.path.exists(path_stations):
            print 'data already exists, continuing ...'
            continue

        #---- prepare redownload ----
        tstart = event.origin+event.ctime+t1
        tend   = event.origin+event.ctime+t2
        downloadlist = [('*',stat,'*','LHZ',tstart,tend) for stat in unique_stations]

        print 'downloading ...'.format(tstart,tend)
        try:
            inventory = client.get_stations_bulk(downloadlist,level='response')
            events    = client.get_events(minmag=5.0,starttime=tstart,endtime=tend)
            waveforms = client.get_waveforms_bulk(downloadlist,
                                              attach_response=False,longestonly=True)
        except Exception,err:
            print err
            print 'ERROR WHILE DOWNLOADING'
            continue

        #---- write to file ----
        try:
            inventory.write(path_stations,format='STATIONXML')
            events.write(path_events,format='QUAKEML')
            waveforms.write(path_mseed,format='MSEED')
        except Exception,err:
            print err
            print 'SAVING ERROR'
            continue

        print '{:d} waveforms written'.format(len(waveforms))

#---- create Directory ----
def create_dir(dirname):
    try:
      os.makedirs(dirname)
      print 'saving under path:',dirname
      return dirname
    except OSError, err:
      if err.errno == 17:
          print '\n directory exists!'
          return dirname
      else:
          print err
          print '\nCan\'t create directory tree'
          raise

#==== SCRIPT EXECUTION ====
if __name__ == "__main__":
    main()
