#!/usr/bin/env python

NUM  = '0123456789'
ABC  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

#==== MAIN FUNCTION ====
def main():
    print cmt2brk('C201312312341A')
    print brk2cmt(cmt2brk('C201302030502A'))

#==== CONVERSION FUNCTIONS ====
def cmt2brk(cmtname):
    #split cmtname into components
    datatype   = cmtname[0]
    century    = cmtname[1:3]
    year       = cmtname[3:5]
    month      = cmtname[5:7]
    day        = cmtname[7:9]
    hour       = cmtname[9:11]
    minute     = cmtname[11:13]
    lastletter = cmtname[13]

    #translation string and table
    brkname = ''

    #datatype and century
    translate = dict( zip(['B19','B20','C19','C20','M19','M20','S19','S20'],ABC[0:8]) )
    brkname += translate[datatype+century]

    #year
    brkname += year
    
    #day
    translate = dict( zip(range(1,32),NUM[1:10]+ABC[0:22]) )
    brkname += translate[int(day)]

    #hour
    translate = dict( zip(range(0,24),NUM[0:10]+ABC[0:14]) )
    brkname += translate[int(hour)]

    #month and minute
    helper   = int(month)*100+int(minute)
    faktor   = helper//35
    remaindr = helper-faktor*35

    translate = dict( zip(range(0,36),NUM[0:10]+ABC[0:26]) )
    brkname += translate[faktor]

    translate = dict( zip(range(0,35),NUM[0:10]+ABC[0:25]) )
    brkname += translate[remaindr]

    #lastletter
    brkname += lastletter

    return brkname

def brk2cmt(brkname):
    #translation string and table
    cmtname = ''

    #datatype and century
    translate = dict( zip(ABC[0:8],['B19','B20','C19','C20','M19','M20','S19','S20']) )
    cmtname += translate[brkname[0]]

    #year
    cmtname += brkname[1:3]

    #month (and minute)
    translate = dict( zip(NUM[0:10]+ABC[0:26],range(0,36)) )
    faktor = translate[brkname[5]]

    translate = dict( zip(NUM[0:10]+ABC[0:25],range(0,35)) )
    remaindr = translate[brkname[6]]

    helper   = faktor*35+remaindr
    month    = helper//100
    minute   = helper - month*100

    cmtname += '%02d'%month
    
    #day
    translate = dict( zip(NUM[1:10]+ABC[0:22],range(1,32)) )
    cmtname += '%02d'%translate[brkname[3]]

    #hour
    translate = dict( zip(NUM[0:10]+ABC[0:14],range(0,24)) )
    cmtname += '%02d'%translate[brkname[4]]

    cmtname += '%02d'%minute

    #lastletter
    cmtname += brkname[7]

    return cmtname

#==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
