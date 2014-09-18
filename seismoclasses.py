import numpy as np
import sys
# local imports:

# PAZ class:
#read plot PAZ (Poles and Zeros) files with instrument response
# plot():
#     import matplotlib.pyplot as plt
#

# STAfile class:
#read plot STATIONS files a.k. SPECFEM, cSEM
# plot():
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.basemap import Basemap
 
# CMTfile class:
#read plot, print CMTSOLUTION files a.k. SPECFEM
# __init__():
#     from obspy.core import UTCDateTime

#OneDModel class: deprecated class. look in yannosclasses for more
#suitable options
#

#mode class: helper class, should be defined in yannosclasses

#manual logarithmic cosTaper function

#plot_grid function. plots lat-lon grid

#xTrace:
#an addon for the obspy trace class, providing a few new methods

#==== MAIN FUNCTION ====
def main():
    ls = np.arange(100)
    lmin = 4
    lmax = 90
    Y = xcosTaper(ls, lmin,lmax)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Y)
    ax.set_xscale('log')
    plt.show()

#==== FUNCTIONS ====
def plot_grid(grid, vrange=None ,label='',fname=None,projection='moll',ending='svg',
              ax=None, colormap=None,lon0=0.,plates=False, lat0=None,cbar=True):
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap, shiftgrid

    nlat, nlon = grid.shape
    size = (7.5,3.5)
    if vrange is None:
        lim = np.max(np.abs(grid))
        vrange = -lim,lim

    if colormap is None:
        colormap = matplotlib.cm.Spectral
    interval = 180./nlat
    latsin = np.arange(-90.+interval/2.,90.0,interval)
    lonsin = np.arange(0.+interval/2.,360.,interval)
    norm = matplotlib.colors.Normalize(vmin=vrange[0], vmax=vrange[1]) 
    if ax==None:
        fig = plt.figure(figsize=size)
    else:
        plt.sca(ax)
        fig = plt.gcf()
    grid_rot,lons = shiftgrid(lon0+180.001,grid,lonsin,start=False)
    if projection=='moll':
        m = Basemap(lon_0 = lon0,resolution='c',area_thresh=10000.,projection='moll')
        x,y = m(*np.meshgrid(lons,latsin))
        p = m.pcolormesh(x,y,grid_rot[::-1,:],norm=norm,cmap=colormap)
        p.set_rasterized(True)
        m.drawcoastlines()
        m.drawmapboundary()
    elif projection=='robin':
        m = Basemap(lon_0 = lon0,resolution='c',projection='robin')
        x,y = m(*np.meshgrid(lons,latsin))
        p = m.pcolormesh(x,y,grid_rot[::-1,:],norm=norm,cmap=colormap)
        p.set_rasterized(True)
        m.drawmapboundary()
        m.drawcoastlines()
    if projection=='ortho':
        m = Basemap(lon_0 = lon0,lat_0=lat0,resolution='c',area_thresh=10000.,projection='ortho')
        x,y = m(*np.meshgrid(lons,latsin))
        p = m.pcolor(x,y,grid_rot[::-1,:],norm=norm,cmap=colormap)
        m.drawcoastlines()
        m.drawmapboundary()
    if projection=='npstere' or projection=='spstere':
        m = Basemap(boundinglat=lat0,lon_0=lon0,resolution='c',projection=projection)
        x,y = m(*np.meshgrid(lons,latsin))
        p = m.pcolormesh(x,y,grid_rot[::-1,:],norm=norm,cmap=colormap)
        m.drawcoastlines()
        p.set_rasterized(True)
        m.drawmapboundary()
    if cbar:
        cbar = m.colorbar(p)
        cbar.set_label(r'$dv_s$')

    def picker(event):
        try:
            lon,lat = m(event.xdata,event.ydata,inverse=True)
            print("lon,lat: ", lon,lat)
        except:
            pass
    cid = fig.canvas.mpl_connect('button_press_event', picker)

    plt.title(label)
    if plates:
        sys.path.append('/home/matthias/projects/python/modules/plates')
        from boundaries import Plates
        plates = Plates()
        bleft = lon0-180.
        bright = lon0+180.
        for name,segment in plates.plate_dict.items():
            color = 'red'
            segment[0,segment[0]>bright] -= 360.
            segment[0,segment[0]<bleft] += 360.
            #find large jumps in longitude
            threshold = 80.
            isplit = np.nonzero(np.abs(np.diff(segment[0])) > threshold)[0]
            subsegs = np.split(segment,isplit+1,axis=1)
            #if len(isplit) > 0:
            #    ipdb.set_trace()

            for seg in subsegs:
                x,y = m(seg[0],seg[1])
                m.plot(x,y,c=color,alpha=0.3)
    if fname is not None:
        fig.savefig(fname)
    return fig

def xcosTaper(ls, band,log=True):
    """
    Returns a cosTaper in logarithmic space
    """
    value = np.zeros_like(ls,dtype=np.float64)
    if log:
        ls = np.log10(ls+1e-10)
        if band[1] > 1:
            c1 = np.log10(band[0]+0.01)
            c2 = np.log10(band[1]+0.01)
        else:
            c1 = -1.
            c2 = -1.
        if band[3] > 1:
            c3 = np.log10(band[2]-0.01)
            c4 = np.log10(band[3]-0.01)
        else:
            c3 = ls.max()+1
            c4 = ls.max()+1
        region1 = np.logical_and(ls>c1,ls<c2)
        region2 = np.logical_and(ls>c2,ls<c3)
        region3 = np.logical_and(ls>c3,ls<c4)
        value[region1] = 0.5*(1.-np.cos(np.pi*(ls[region1]-c1)/(c2-c1)))
        value[region2] = 1.
        value[region3] = 0.5*(1.+np.cos(np.pi*(ls[region3]-c3)/(c4-c3)))
    return value

#==== CLASSES ====
class OneDModel:
    """
    this class reads a 1D model ascii mode file from yannos and uses it to
    extract the Q, as well as the group velocity value of fundamental mode
    surface waves (possibly overtones).
    """
    def __init__(self, fn_yannos, fn_model=None):
        from operator import itemgetter
        self.maxf = 10e-3
        self.fname = fn_yannos

        modefile = [line.strip().rsplit(None,7) for line in open(fn_yannos).readlines()[17:]]
        modefile = sorted(modefile,key=itemgetter(2))

        Ls = []
        Fs = []
        Qs = []
        Vs = []

        for mode in modefile:
            n = int(mode[0].split()[0])
            l = int(mode[0].split()[2])
            w = float(mode[2])*1e-3
            v = float(mode[4])
            Q = float(mode[5])
            if n == 0 and l>5:
                Ls.append(l)
                Fs.append(w)
                Qs.append(Q)
                Vs.append(v)

        self.Ls = np.array(Ls)
        self.Fs = np.array(Fs)
        self.Qs = np.array(Qs)
        self.Vs = np.array(Vs)

        if fn_model:
            file_object = open(fn_model,'r')
            self.name    = file_object.readline()
            self.flags   = file_object.readline().split()
            self.nlayers = file_object.readline().split()
            self.data    = np.loadtxt(file_object)
            self.labels  = np.array(['radius', 'density', 'vpv', 'vsv', 'Qk', 'Qmu', 'vph', 'vsh', 
                                                                                            'eta'])
            self.units  = np.array(['m','kg/m^3','m/s','m/s','m/s','-','-','m/s','m/s','-'])

    def get(self, variable):
        """
        :param variable: available quantities are 'density','vpv','vsv','Qk','Qmu','vph','vsh','eta'
        :returns: numpy array of the desired variable
        """
        assert variable in self.labels, 'could not find quantity: ' + variable
        icolumn = np.where(self.labels == variable)[0][0] #index of first loc of variable in labels
        return np.copy(self.data[:,icolumn])

    def set(self, variable, array):
        """
        :param variable: available quantities are 'density','vpv','vsv','Qk','Qmu','vph','vsh','eta'
        :param array:    the new array (should have correct dimensions)
        """
        assert variable in self.labels, 'could not find quantity: ' + variable
        icolumn = np.where(self.labels == variable)[0][0] #index of first loc of variable in labels
        self.data[:,icolumn] = np.copy(array)

    def getQ(self,fmeas):
        return np.interp(fmeas,self.Fs,self.Qs)

    def getV(self,fmeas):
        return np.interp(fmeas,self.Fs,self.Vs)

    def plotQ(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('1D model Attenuation from file: %s'%self.fname.split('/')[-1])
        ax.set_xlabel('frequency in [Hz]')
        ax.set_ylabel('Attenuation factor Q [unitless]')
        ax.plot(self.Fs,self.Qs)

    def plotV(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('1D model group velocity curve from file: %s'%self.fname.split('/')[-1])
        ax.set_xlabel('frequency in [Hz]')
        ax.set_ylabel('group velocity in [km/s]')
        ax.plot(self.Fs,self.Vs)
      
    def arrival_times(self, statdist, npfreqs, narr = 10):
        """
        this function computes the surface wave arrival times for a given 1dmodel
        :param statdist: station distance in km
        :param npfreqs:  numpy array with frequencies (linear interpolation between modes)
        :param narr: number of arrivals
        :param fmin: minimum frequency calculated
        :param fmax: maximum frequency calculated

        """
        npvs = self.getV(npfreqs)
        nparrivals = np.zeros( (len(npfreqs),narr) )

        dt_arr = 2. * np.pi * 6371./npvs
        dt_sta = statdist/npvs
    
        for i in range(0,narr/2,1):
            minor =  dt_sta + dt_arr * float(i)
            major = -dt_sta + dt_arr * float(i+1)
            nparrivals[:,i*2] = minor
            nparrivals[:,i*2+1] = major
        
        return nparrivals

    @staticmethod
    def readmodes(fname, branch=0):
        """
        returns a list of modes and radii on which they are defined
        :param fname: filename of yannos binary file
        :returns r: radii of layers in yannos binary
        :returns modelist: list of mode objects
        """
        offset = 0
        #--- read general info (number of modes, number of layers, radii of layers) ---
        binfile = open(fname,'rb')
        pad,nbceau,nbcou_lay = np.fromfile(binfile,count=3,dtype=np.int32)
        offset += 3 * 4 #pad, nbceau, nbcou_lay (3xint)
        binfile.seek(offset)
        r = np.fromfile(binfile,count=222,dtype=np.float32)
        offset += 222 * 4 #222 radii
        binfile.seek(offset)
        pad = np.fromfile(binfile,count=1,dtype=np.int32)
        offset += 1 * 4 #pad
        binfile.seek(offset)
    
        modelist = []
        #--- read modes ---
        for imode in np.arange(3000):
            eigenfct = []
            pad,n,l = np.fromfile(binfile,count=3,dtype=np.int32)
            offset += 3 * 4 #pad,n,l (integer)
            binfile.seek(offset)
            w,q,gv = np.fromfile(binfile,count=3,dtype=np.float64)
            offset += 3 * 8 #w,q,gv (double precision)
            binfile.seek(offset)
            eigenfct.append(np.fromfile(binfile,count=nbcou_lay,dtype=np.float64))
            offset += nbcou_lay * 8 # u (double precision)
            binfile.seek(offset)
            eigenfct.append(np.fromfile(binfile,count=nbcou_lay,dtype=np.float64))
            offset += nbcou_lay * 8 # du
            binfile.seek(offset)
            if l != 0:
                modetype = 'spheroidal'
                eigenfct.append(np.fromfile(binfile,count=nbcou_lay,dtype=np.float64))
                offset += nbcou_lay * 8 # v
                binfile.seek(offset)
                eigenfct.append(np.fromfile(binfile,count=nbcou_lay,dtype=np.float64))
                offset += nbcou_lay * 8 # dv
                binfile.seek(offset)
                eigenfct.append(np.fromfile(binfile,count=nbcou_lay,dtype=np.float64))
                offset += nbcou_lay * 8 # phi
                binfile.seek(offset)
                eigenfct.append(np.fromfile(binfile,count=nbcou_lay,dtype=np.float64))
                offset += nbcou_lay * 8 # dphi
                binfile.seek(offset)
            else:
                modetype = 'radial'
    
            np.fromfile(binfile,count=1,dtype=np.int32)
            offset += 1 * 4 # pad
            binfile.seek(offset)
            buf = np.array([])
            for elem in eigenfct:
                buf = np.append(buf,elem)
            if n == branch: #for fundamental mode only
                modelist.append(mode(imode,nbcou_lay,modetype,r,n,l,w,q,gv,buf))
        return modelist

#--------------- MODE CLASS --------------
class mode:
    def __init__(self,i,nbcou_lay,modetype,radii,n,l,w,q,gv,buf):
        self.index = i
        self.modetype = modetype
        self.radii = radii
        self.n = n
        self.l = l
        self.omega = w
        self.q = q
        self.gv= gv
        self.k = np.sqrt(l*(l+1))
        #data arrays
        self.u = buf[0*nbcou_lay:1*nbcou_lay]
        self.du = buf[1*nbcou_lay:2*nbcou_lay]
        self.v = 0.0
        self.dv = 0.0
        self.p = 0.0
        self.dp = 0.0
        self.w = 0.0
        self.dw = 0.0
        if modetype == 'spheroidal':
            self.v = buf[2*nbcou_lay:3*nbcou_lay]
            self.dv = buf[3*nbcou_lay:4*nbcou_lay]
            self.p = buf[4*nbcou_lay:5*nbcou_lay]
            self.dp = buf[5*nbcou_lay:6*nbcou_lay]

    def pt(self):
        print 'Mode No. %1d: %1dS%1d f=%2.2e'%(self.index,self.n,self.l,self.omega/2./np.pi)

    def get_kernels(self,rho,vp,vs):
        """computes vp/vs kernels according to Dahlen & Tromp Eq: 9.13 ff."""
        K_kappa = (self.radii * self.du + 2.*self.u - self.k * self.v)**2/(2.*self.omega)

        K_mu    = 1./3. * (2.*self.radii * self.du - 2. * self.u + self.k * self.v)**2 + \
                  (self.radii * self.dv - self.v + self.k * self.u)**2 + \
                  (self.radii * self.dw - self.w)**2 + \
                  (self.k**2 - 2) * (self.v**2 + self.w**2)/(2.*self.omega)

        K_alpha = 2. * rho * vp * K_kappa
        K_beta  = 2. * rho * vs * (K_mu - 4./3. * K_kappa)
        return K_alpha, K_beta

#----------------------------------
from obspy.core import read, Trace
from UCBFilter import Filter
class xTrace(Trace):
    def __init__(self, fname): 
        """
        This is a subclass of the obspy trace which is compatible with all obspy functions
        but adds several advanced features
        """
        self.__dict__ = read(fname)[0].__dict__

    def get_timeaxis(self):
        """returns a numpy array with the time axis in seconds. Zero is the event time."""
        #get information from sac header
        start = self.stats.sac.b
        dt = self.stats.delta
        npts = self.stats.npts
        time = np.linspace(0., npts*dt, npts) + start
        return time

    def bandpass(self,freqband = (1e-3,2e-3,10e-3,12e-3)):
        UCBf = Filter(np.array(freqband) * 2.*np.pi)
        self.data = UCBf.apply(self.data,self.stats.delta,taper_len = 0.0)

    def increase_delta(self, new_delta=10.0):
        """calls the original decimate function but takes delta as argument instead of a factor"""
        print 'desired decimation factor: ',new_delta/self.stats.delta
        if self.stats.delta > new_delta:
            return
        factor = min( int(new_delta/self.stats.delta), 15)
        print 'decimating trace by a factor of:',factor,'(max 15)'
        self.decimate(factor, no_filter=False)

    def getrfft(self, taper=None):
        """returns the frequency axis and the complex spectrum of the trace"""
        if taper == 'multitaper':
            raise NotImplementedError("multitaper method is not yet implemented")
        elif taper == 'hanning':
            print 'applying taper:',taper
            data = self.data * np.hanning(self.stats.npts)
        else:
            data = self.data
        nfreq = self.stats.npts
        freq = np.arange(0,nfreq+1,2)/2/float(nfreq*self.stats.delta)
        fft = np.fft.rfft(data,n=nfreq)*self.stats.delta/float(nfreq)
        return freq,fft

    def resample_to(self, target_trace, filter=False):
        import scipy.interpolate as interp
        """matches timeaxis to obs_trace. This should be done after filtering"""
        #first try to decimate further
        samp_self = self.stats.sampling_rate
        samp_target = target_trace.stats.sampling_rate
        fac = np.floor(self.stats.sampling_rate/target_trace.stats.sampling_rate).astype(int)
        while fac>1:
            print 'downsampling trace by a factor of:',fac
            self.decimate(fac,no_filter=True)
            fac = np.floor(self.stats.sampling_rate/samp_target).astype(int)
        print 'sampling changed from ',samp_self,'to',self.stats.sampling_rate,\
              '(matching: %f)'%samp_target
        samp_self = self.stats.sampling_rate
        #interpolate to target trace
        target_time = target_trace.get_timeaxis()
        timeshift = self.stats.starttime - target_trace.stats.starttime
        Interpolator = interp.InterpolatedUnivariateSpline(self.get_timeaxis()+timeshift,self.data)
        self.data = Interpolator(target_time)
        #update meta
        self.stats.sampling_rate = samp_target
        self.stats.sac.b = target_trace.stats.sac.b
        self.stats.starttime = target_trace.stats.starttime
        assert (self.stats.npts == target_trace.stats.npts),\
                'number of points differs after matching'

    def xplot(self):
        """custom plot function"""
        import matplotlib.pyplot as plt
        plt.figure()
        time = self.get_timeaxis()
        plt.plot(time,self.data)

    def convolve_stf(self,hdur):
        #gaussian stf:
        def source_time_function(dt,hdur):
            length = 5.*hdur
            npoints = length*2/dt
            times = np.linspace(-length,length,npoints)
            alpha = 1.628
            norm = alpha/(np.sqrt(np.pi)*hdur)
            return dt * norm * np.exp(-(alpha/hdur * times)**2)

        stf = source_time_function(self.stats.delta,hdur)
        convolved_data = np.convolve(stf,self.data,'same')
        self.data = convolved_data
        #plotting functions to examine the spectrum
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(self.get_timeaxis(),self.data,label='original')
        #plt.plot(self.get_timeaxis(),convolved_data,label='convolved')
        #plt.legend()
        #nfreq = self.stats.npts
        #freq = np.arange(0,nfreq+1,2)/2/float(nfreq*self.stats.delta)
        #fft_orig = np.fft.rfft(self.data,n=nfreq)*self.stats.delta/float(nfreq)
        #fft_conv = np.fft.rfft(convolved_data,n=nfreq)*self.stats.delta/float(nfreq)
        #plt.figure()
        #plt.plot(freq,np.abs(fft_orig),label='original')
        #plt.plot(freq,np.abs(fft_conv),label='convolved')
        #plt.legend()
        #plt.show()


#----------------------------------
class PAZfile():
    """
    This is a simple class that reads and stores a SAC poles and zeros
    file. It can plot its impulse response as well as its frequency
    characteristics.
    """
    def __init__(self,fname):
        pazFile = open(fname).readlines()
        #read meta info
        meta_lines = [tuple(line.strip('*\n ').split(':',1)) for line in pazFile \
                                                             if len(line.split(':')) > 1]
        self.stats = dict([(att.strip().split()[0],val.strip()) for att,val in meta_lines])
        
        iZeros = [index for index, line in enumerate(pazFile) if 'ZEROS' in line][0]
        self.NZeros = int(pazFile[iZeros].strip().split()[-1])
        iPoles = iZeros + 1 + self.NZeros
        self.NPoles = int(pazFile[iPoles].strip().split()[-1])
        print 'Number of Zeros: %d, Number of Poles: %d'%(self.NZeros,self.NPoles)
        self.zeros = [ float(l.strip().split()[0]) + 1j * float(l.strip().split()[1]) 
                                               for l in pazFile[iZeros+1:iPoles] ]
        self.poles = [ float(l.strip().split()[0]) + 1j * float(l.strip().split()[1]) 
                                               for l in pazFile[iPoles+1:iPoles+ 1 + self.NPoles] ]
        self.f,self.H = self.freqResp()
        self.t,self.D = self.impResp()

    def freqResp(self):
        print 'calculating frequency response ...'
        length = 4. * 3600.
        N = 2**15
        dt = length/N
        f = np.logspace(-4,1,2**10)
        df = f[1] - f[0]
        s = 2.j * np.pi * f
        Nom = np.ones_like(s)
        DeNom = np.ones_like(s)
        for z in self.zeros[:]: Nom *= (s - z)
        for p in self.poles: DeNom *= (s - p)
        H = Nom/DeNom
        return f,H

    def impResp(self):
        t = np.linspace(-0.5,0.5,2**12) * 3600.
        mask = t > 0
        D = np.zeros_like(t)
        for p in self.poles:
            D[mask] += np.real(np.exp(p * t[mask]))
        D = np.convolve(D,np.hanning(300),'same')
        return t,D

    #---- plotting ----
    def plot(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.plot(self.f,np.abs(self.H))
        ax2.plot(self.t,self.D)
        if hasattr(sys,'ps1'):
            plt.show()

#---- class for a STATIONS file as used by SPECFEM ----
class STAfile:
    def __init__(self,fname=None,fmt='specfem'):
        #stat_dict[key] = tuple(0 = Network, 1 = lat, 2 = lon, 3 = elevation, 4 = burial)
        if not fname:
            self._stat_dict = dict()
            self.Nstat = len(self._stat_dict)
        elif fmt=='specfem':
            self._stat_dict = dict([(l.strip().split()[0],l.strip().split()[1:]) \
                                                                     for l in open(fname,'r')])
            self.Nstat = len(self._stat_dict)
        elif fmt =='acal':
            self._stat_dict = dict([(l.strip().split()[0],['NaN'] + l.strip().split()[1:-1]+\
                                           [0.0,0.0]) for l in open(fname,'r').readlines()[4:]])
            self.Nstat = len(self._stat_dict)

    def __getitem__(self,key):
        return self._stat_dict[key]

    def getall(self):
        keys,values = zip(*self._stat_dict.items())
        return keys, values

    def get_coords(self):
        ntws,lats,lons,elevs,burials = zip(*self._stat_dict.values())
        return np.array([lats,lons],dtype=np.float32)

    def add_station(self, name, network, lat, lon, elevation, burial):
        if name in self._stat_dict:
            print 'station %s already exists, choose different name!'%name
            print self._stat_dict[name]
        else:
            self._stat_dict[name] = (network, lat, lon, elevation, burial)

    def get_latlon(self,stat_name):
        lat = float(self._stat_dict[stat_name][1])
        lon = float(self._stat_dict[stat_name][2])
        return lat,lon

    def pprint(self):
        print 'Number of Stations: ', len(self._stat_dict)

    def write(self,fname,fmt):
        if fmt == 'acal':
            outfile = open(fname,'w')
            outfile.write('#number of receivers\n%d\n\n'%self.Nstat)
            outfile.write('# <name> <lat> <lon>\n')
            for station, val in self._stat_dict.items():
                lat = float(val[1])
                lon = float(val[2])
                outfile.write('%s %f %f\n'%(station[:4],lat,lon))
            outfile.close()
        elif fmt == 'specfem':
            outfile = open(fname,'w')
            for station, val in self._stat_dict.items():
                ntw = val[0]
                lat = float(val[1])
                lon = float(val[2])
                ele = val[3]
                bur = val[4]
                outfile.write('%4s     '%station + '%4s     % 6.2f % 6.2f % 6.2f % 6.2f\n'%(ntw,
                                                                                 lat,lon,ele,bur))
            outfile.close()
        else:
            print 'format %s unknown'%fmt

    def plot(self,fname=None):
        """
        plots a map with the stations. if you want to write the map to a file, specify the keyword
        fname = "filename".
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        fig = plt.figure(figsize = (12,8))
        m = Basemap(llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,\
                    resolution='c',area_thresh=10000.,projection='cyl')

        plt.title('%d stations'%len(self._stat_dict))
        m.drawcoastlines()
        m.fillcontinents()
        for i,(stat,val) in enumerate( self._stat_dict.items() ):
            if i%10 == 0: print '%d/%d done'%(i,self.Nstat)
            latstat = float(val[1])
            lonstat = float(val[2])
            x,y = m(lonstat,latstat)
            plt.plot(x,y,'o',c='g')
        if not fname == None:
            fig.savefig(fname)
        elif hasattr(sys,'ps1'): 
            plt.show()

    def plot_source_station(self,station_name,evla,evlo,fname=None):
        """
        plots a map with a path from the source to the station. if you want to write the map to a 
        file, specify the keyword fname = "filename".
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        fig = plt.figure(figsize = (12,8))
        m = Basemap(llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,\
                    resolution='c',area_thresh=10000.,projection='cyl')
        plt.title('%d stations'%len(self._stat_dict))
        m.drawcoastlines()
        m.fillcontinents()
        stla,stlo = self.get_latlon(station_name)
        latstat = float(stla)
        lonstat = float(stlo)
        x,y = m(lonstat,latstat)
        plt.plot(x,y,'o',c='g')
        if not fname == None:
            fig.savefig(fname)
        elif hasattr(sys,'ps1'): 
            plt.show()

#---- class for a CMTSOLUTION file as used by SPECFEM ----
class CMTfile:
    def __init__(self,fname,fmt='specfem'):
        from obspy.core import UTCDateTime
        infile = open(fname)
        if fmt == 'specfem':
    
            head = infile.readline().strip()
            self._catalog = head[:4]
            self._time = UTCDateTime(head[4:27])
            self._magb  = float(head[52:55])
            self._mags  = float(head[55:60])
            self._area = head[60:]
    
            data = [l.strip().split(':') for l in infile]
            self._name      = data[0][1].strip()
            self._timeshift = float(data[1][1])
            self._hdur      = float(data[2][1])
            self._lat       = float(data[3][1])
            self._lon       = float(data[4][1])
            self._dep       = float(data[5][1])*1e3 #in km in Specfem, we go to SI units
            self._mtensor   = tuple([float(l[1])*1e-7 for l in data[6:]]) 
                                                  #specfem is in dyne-cm, we convert it to N-m
        elif fmt == 'acal':
            self._catalog = 'NA'
            self._time = None
            self._magb  = 0.0
            self._mags  = 0.0
            self._area = 'NA'
    
            data = open(fname).readlines()[7].split()
            self._name      = data[0]
            self._timeshift = float(data[10])
            self._hdur      = 0.0 #data is bandpassed but typically with a flat top within the band
            self._lat       = float(data[8])
            self._lon       = float(data[9])
            self._dep       = float(data[7]) #in km in Specfem, we go to SI units
            self._mtensor   = tuple([float(l) for l in data[1:7]]) #specfem is in dyne-cm, we 
                                                                   #convert it to N-m
        infile.close()

    def get_latlon(self):
        return self._lat, self._lon

    def get_time(self):
        return self._time

    def get_timestamp(self):
        return self._time.timestamp

    def write(self,fname,fmt):
        if fmt == 'acal':
            outfile = open(fname,'w')
            outfile.write('# source file\n\n#number of sources\n')
            outfile.write('1\n\n')
            outfile.write('# specification of sources\n'+\
            '# <name> <Mrr> <Mtt> <Mpp> <Mrt> <Mrp> <Mtp> <depth> <lat> <lon> <t_start> <t_end>\n')
            outfile.write('%s '%self._name[:8] + '%e %e %e %e %e %e '%self._mtensor +
                          '%f %f %f %d %d'%(self._dep,self._lat,self._lon,0,30*3600))
        else:
            print 'format %s unknown'%fmt


    #---- printing ----
    def pprint(self):
        print 'catalog: ',self._catalog
        print 'area: ',self._area
        print 'magb, mags: ',self._magb, self._mags
        print 'time: ',self._time
        print 'name: ',self._name
        print 'location (lon,lat): ',self._lon,' ',self._lat
        print 'depth: ',self._dep
        print 'moment tensor: ',self._mtensor

    #---- could add plot on map ----

#==== EXECUTION ====
if __name__=="__main__":
    main()
