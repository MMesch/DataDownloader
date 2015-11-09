#!/usr/bin/env python
"""
this script redownloads waveform data from iris, etc to get an idea about the pre-event
noise level of each data trace from SEMUCB
"""
from obspy.core import read,Stream
from obspy.fdsn.client import Client
from ndk_rec_file import NDKFile,RecFile
from scipy.interpolate import LSQUnivariateSpline
import argparse
import numpy as np
import os

#==== MAIN FUNCTION ====
def main():
    #==== INITIALIZE ====
    #---- parameters ----
    redownload = True
    basedir = 'semucb_dataset'

    #---- input files ----
    fname_evcatalogue = os.path.join(basedir,'evcatalogue_semucb.ndk')
    fname_stcatalogue = os.path.join(basedir,'receivers.dat')

    #==== PREPARE DOWNLOAD LIST ====
    print 'reading station list...'
    stcatalogue = RecFile(fname_stcatalogue)
    print 'retrieved {:d} stations'.format(stcatalogue.ndata)

    print 'reading event catalogue...'
    evcatalogue = NDKFile(fname_evcatalogue)
    print 'found {:d} events'.format(evcatalogue.nevents)

    #==== LOOP THROUGH EVENTS ====
    for name,event in zip(evcatalogue.names,evcatalogue.events):
        print '---- {:s} ----'.format(name)


#==== MAKE OUTPUT FILE STRUCTURE ====
def make_dir_out(evname):
    """creates the necessary output folder and files"""
    #make output directory
    dir_out = os.path.join('data','plots',evname)
    try:
      os.makedirs(dir_out)
      print 'saving under path:',dir_out
      return dir_out
    except OSError, err:
      if err.errno == 17:
          print '\ndirectory exists!'
          return dir_out
      else:
          print err
          print '\nCan\'t create directory tree'
          raise
    return dir_out

#==== SCRIPT EXECUTION ====
if __name__ == "__main__":
    main()
