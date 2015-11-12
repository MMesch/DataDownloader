#!/usr/bin/env python
"""
This script removes the instrument response function from the raw data of the
semucb dataset
"""

from obspy import read,read_inventory, readEvents
from ndk_rec_file import NDKFile
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import os

#==== MAIN FUNCTION ====
def main():
    reprocess  = True
    basedir    = 'semucb_dataset'
    datadir    = os.path.join(basedir,'data')

    #---- input files ----
    fname_evcatalogue = os.path.join(basedir,'evcatalogue_semucb.ndk')

    #==== PROCESSING OPTIONS ====
    fs          = 1./8.
    pre_filt    = (0.,1./3600.,1./2.*fs-0.0001,1./2.*fs)
    water_level = 100

    #==== PREPARE STATION/EVENT LIST ====
    print 'reading event catalogue...'
    evcatalogue = NDKFile(fname_evcatalogue)
    print 'found {:d} events'.format(evcatalogue.nevents)

    #==== LOOP THROUGH DATA ====
    for name in evcatalogue.names:
        print '---- {:s} ----'.format(name)
        #check if data is already processed
        path      = os.path.join(datadir,name)
        fname_out = path + '_wfproc.mseed'
        if os.path.exists(fname_out) and not reprocess:
            print 'already processed'
            continue

        #read dataset:
        try:
            events    = readEvents(path + '_events.xml')
            stations  = read_inventory(path + '_stations.xml',format='STATIONXML')
            waveforms = read(path+'_waveforms.mseed',format='MSEED') 
        except Exception,err:
            print err
            print 'can\'t open dataset'
            continue

        #remove tides with splines and then the instrument response
        waveforms.attach_response(stations)
        nwaveforms = len(waveforms)
        for itr, tr in enumerate(waveforms):
            if itr%20==0: print 'removing response {:d}/{:d}'.format(itr,nwaveforms)
            #detide with spline (reduces boundary effects)
            trange   = tr.stats.endtime - tr.stats.starttime
            taxis    = tr.times()
            dspline  = 500.
            nsplines = trange/dspline
            splknots = np.linspace(taxis[0]+dspline/2.,taxis[-1]-dspline/2.,nsplines)
            spl      = LSQUnivariateSpline(taxis,tr.data,splknots)
            tr.data -= spl(taxis).astype('int32')
            try:
                tr.remove_response(pre_filt=pre_filt,output='ACC',water_level=water_level,
                                                            taper=True,taper_fraction=0.05)
            except Exception,err:
                print err
            tr.data = tr.data.astype(np.float32)

        #save waveforms to file
        waveforms.write(fname_out,format='MSEED',encoding='FLOAT32')



#==== SCRIPT EXECUTION ====
if __name__ == "__main__":
    main()

