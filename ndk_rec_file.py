from obspy.core import UTCDateTime
import re
from brk2cmt import brk2cmt

#==== NDK Event Catalogue ====
class NDKFile(object):
    def __init__(self,fname):
        print 'reading data file...'
        lines = open(fname,'r').readlines()
        print 'extracting dates...'
        blocks = [lines[i:i+5] for i in range(0,len(lines),5)]
        self.events = []
        self.names  = []
        for block in blocks:
            event = NDKEvent(block)
            self.events.append(event)
            self.names.append(event.cmtid)
        self.nevents = len(self.events)

    def findname(self,name):
        try:
            ind = self.names.index(name)
        except ValueError,err:
            print err
            cmtname = brk2cmt(name)
            print 'couldn\'t find 8 char event: %s'%name
            print 'trying 14 char name: %s'%cmtname
            ind = self.names.index(cmtname)
        return self.events[ind]

class NDKEvent(object):
    def __init__(self,ndkblock):
        #save full info
        self.block = ''.join(ndkblock)

        #extract specific information:
        timestr = ndkblock[0][5:27]
        fmt = re.compile( '(\d*)/(\d*)/(\d*) (\d*):(\d*):(\d*).\d' )
        match = re.match(fmt,timestr)
        if match is None:
            self.origin = UTCDateTime()
        else:
            ye,mo,da,ho,mi,se = [int(ma) for ma in match.groups()]
            ho=ho%23
            mi=mi%59
            se=se%59
            self.origin = UTCDateTime(ye,mo,da,ho,mi,se)

        self.position = float(ndkblock[0][27:34]),float(ndkblock[0][34:42])
        self.depth    = float(ndkblock[0][43:48])
        self.magb     = float(ndkblock[0][48:52])
        self.mags     = float(ndkblock[0][52:56])
        self.region   = ndkblock[0][56:]

        #inversion information:
        self.cmtid     = ndkblock[1][:15].strip()

        #centroid information:
        self.ctime     = float(ndkblock[2][14:19])
        self.cposition = float(ndkblock[2][23:30]),float(ndkblock[2][35:43])
        self.cdepth    = float(ndkblock[2][49:54])

        #centroid information:
        exp = 10**float(ndkblock[3][0:3])
        self.focal_mechanism = [float(num)*exp for num in ndkblock[3][3:].split()[::2]]

    def info(self):
        print self.__dict__

class RecFile(object):
    def __init__(self,fname):
        self.stations  = []
        self.networks  = []
        self.positions = []
        fobject = open(fname,'r')
        h1 = fobject.readline()
        h2 = fobject.readline()
        h3 = fobject.readline()
        for line in fobject:
            name,lat,lon,weight = line.split()
            self.stations.append(name.split('.')[1].replace('_','').upper())
            self.networks.append(name.split('.')[0].replace('_','').upper())
            self.positions.append( (float(lat),float(lon)) )

        self.ndata = len(self.stations)

    def get_position(self,station,network):
        semid = (station).upper()
        try:
            ind = self.stations.index(semid)
            return self.positions[ind]
        except ValueError,err:
            print err
            print 'can\'nt find station'
            return 0.,0.
