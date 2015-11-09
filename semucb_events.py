#!/usr/bin/env python

from ndk_rec_file import NDKFile

#==== MAIN FUNCTION ====
def main():
    #file names
    fname_evcatalogue = 'cmtcatalogue/evcatalogue_jan76_dec13.ndk'
    fname_evssemucb   = 'semucb_dataset/evlist'
    fname_semucbndk   = 'semucb_dataset/evcatalogue_semucb.ndk'

    #read whole cmt catalogue
    evcatalogue       = NDKFile(fname_evcatalogue)

    #open semucb eventlist
    semucb_evlist = open(fname_evssemucb,'r')

    #loop through semucb event names and write cmt information to file
    semucbndk = open(fname_semucbndk,'w')
    for line in semucb_evlist:
        event = evcatalogue.findname(line.strip())
        semucbndk.write(event.block)

    semucb_evlist.close()
    semucbndk.close()

#==== SCRIPT EXECUTION ====
if __name__ == "__main__":
    main()
