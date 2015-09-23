#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
from obspy import read, read_inventory

def main():
    args = readArguments()
    waveforms = read(args.mseed,format='MSEED')
    inventory = read_inventory(args.stationxml,format='STATIONXML')
    waveforms.attach_response(inventory)
    pre_filt = (1./400.,1./300.,1./50.,1./30.)
    waveforms.remove_response(output='DISP',pre_filt=pre_filt)
    waveforms.plot(outfile='out.png')

    #plt.show()

#-----------
def readArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('mseed',help='mseed file with waveforms')
    parser.add_argument('stationxml',help='stationxml file with response information')
    args = parser.parse_args()
    return args

#==== EXECUTE =====
if __name__ == "__main__":
    main()
