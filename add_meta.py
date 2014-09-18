#!/usr/bin/env python
"""This script adds meta information to SAC files from associated PAZ files"""
import os
import argparse
from obspy.sac import SacIO
from seismoclasses import CMTfile, PAZfile
from obspy.core.util import gps2DistAzimuth
import glob

#==== MAIN FUNCTION ====
def main():
    args = readArguments()
    fnames = searchData(args.data_folder)
    for data_fname, paz_fname in zip(*fnames):
        add_trace_meta(data_fname, paz_fname, cmtfile=args.cmtfile)

#---------------
def add_trace_meta(data_fname, paz_fname, cmtfile=None):
    sacfile = SacIO(data_fname,headonly=True)
    paz = PAZfile(paz_fname)

    stla = float(paz.stats['LATITUDE'])
    stlo = float(paz.stats['LONGITUDE'])
    inst = [word for word in paz.stats['INSTTYPE'].split() if ('STS' in word)]
    if len(inst)>0: inst = inst[0]
    else: inst = 'not found'
    sacfile.kinst = inst
    sacfile.knetwk = paz.stats['NETWORK']
    sacfile.kstnm = paz.stats['STATION']
    sacfile.khole = paz.stats['LOCATION']
    sacfile.kcmpnm = paz.stats['CHANNEL']
    sacfile.stla = stla
    sacfile.stlo = stlo
    sacfile.stel = paz.stats['ELEVATION']
    sacfile.stdp = paz.stats['DEPTH']
    if not cmtfile == None:
        cmt = CMTfile(cmtfile)
        evla, evlo = cmt.get_latlon()
        evtime = cmt.get_time() #this is an obspy UTCDateTime object
        sacfile.evla = evla
        sacfile.evlo = evlo
        sacfile.evdp = cmt._dep
        sacfile.mag  = cmt._mags
        #change begin marker
        sacfile.b = float(sacfile.starttime - evtime)
        #update starttime to event time
        sacfile.nzyear = evtime.year
        sacfile.nzjday = evtime.julday
        sacfile.nzhour = evtime.hour
        sacfile.nzmin  = evtime.minute
        sacfile.nzsec  = evtime.second
        sacfile.nzmsec = evtime.microsecond
        distance, az, baz = gps2DistAzimuth(evla, evlo, stla, stlo)
        sacfile.dist = distance*1e3 #[m] -> [km]
        sacfile.az   = az
        sacfile.baz  = baz
        sacfile._get_date()

    sacfile.WriteSacHeader(data_fname)
    
#-----------
def searchData(base_dir,filter_expr='*'):
    data_dir = base_dir + '/Waveforms/'
    resp_dir = base_dir + '/PAZ/'
    data_fnames = glob.glob(data_dir + '/'+filter_expr + '.noi')
    paz_fnames  = [resp_dir + fname.rsplit('/')[-1].replace('.sac.noi','.paz') for fname in data_fnames]
    return data_fnames,paz_fnames

#-----------
def readArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder',help='folder with subfolders /Waveforms/ /PAZ/')
    parser.add_argument('--filenames',help='filters files with this expression (e.g. *.VHZ.*)', default='*.VHZ.*')
    parser.add_argument('--cmtfile',help='CMTSOLUTION file', default=None)
    args = parser.parse_args()
    #check that subfolders exists
    if not os.path.isdir(args.data_folder + '/Waveforms') or \
       not os.path.isdir(args.data_folder + '/PAZ'):
        raise Exception("folder doesn't contain a Waveforms/ and a PAZ/ subdirectory")
    return args

#==== EXECUTE ====
if __name__== "__main__":
    main()
