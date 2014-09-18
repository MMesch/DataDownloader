#!/usr/bin/env python
import argparse
import glob
import os
import subprocess

def main():
    """
    This script deconvolves the instrument response using SAC and a PAZ file. Also it updates header
    information with station parameters from the PAZ file and optionally with a CMTSOLUTION file.
    """
    args = readArguments()
    fnames = searchData(args.data_folder)
    for data_fname, paz_fname in zip(*fnames):
        deconvolveResponse(data_fname, paz_fname)

#------------
def deconvolveResponse(data_fname, paz_fname, bandpass = (1e-3,2e-3,50e-3,55e-3)):
    p = subprocess.Popen(['/home/matthias/programs/sac/bin/sac'],
                         stdout = subprocess.PIPE,
                         stdin  = subprocess.PIPE,
                         stderr = subprocess.STDOUT )

    #--- sac commands below ---
    freqlims = '%s %s %s %s'%bandpass
    s = \
    'setbb pzfile ' + paz_fname + '\n' + \
    'read ' + data_fname + '\n' + \
    'rtrend' + '\n' + \
    'taper' + '\n' + \
    'trans from polezero s %pzfile to NONE freqlim ' + freqlims + '\n' + \
    'write ' + data_fname + '.noi\n'
    'quit\n'
    #--- launch sac and retrieve output ---
    sacout,sacerr = p.communicate(s)

    #--- write output and print it to screen synchronous because of threading ---
    output =  '\nstarting sac with files: %s and %s'%(data_fname, paz_fname)
    output = 'sac command used:\n' + s
    output += '\nSAC OUTPUT:\n' + sacout 
    output += '\nSAC ERRORS:\n' + str(sacerr)
    output += ' -- Instrument Correction done --\n'
    output += '-----------------------------------'
    print output

#-----------
def searchData(base_dir,filter_expr='*'):
    data_dir = base_dir + '/Waveforms/'
    resp_dir = base_dir + '/PAZ/'
    data_fnames = glob.glob(data_dir + '/'+filter_expr)
    paz_fnames  = [resp_dir + fname.rsplit('/')[-1].replace('.sac','.paz') for fname in data_fnames]
    return data_fnames,paz_fnames

#-----------
def readArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder',help='folder with subfolders /Waveforms/ /PAZ/')
    parser.add_argument('--filenames',help='filters files with this expression (e.g. *.VHZ.*)', default='*.VHZ.*')
    args = parser.parse_args()
    #check that subfolders exists
    if not os.path.isdir(args.data_folder + '/Waveforms') or \
       not os.path.isdir(args.data_folder + '/PAZ'):
        raise Exception("folder doesn't contain a Waveforms/ and a PAZ/ subdirectory")
    return args

#==== EXECUTE =====
if __name__ == "__main__":
    main()
