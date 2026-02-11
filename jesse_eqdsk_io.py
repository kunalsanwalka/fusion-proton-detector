#SF reworked 2023/11 to use more standard eqdsk formatting and fix issues with
#vessel and limiter writing
import sys
import numpy as np
import matplotlib.pyplot as plt
from pleiades.analysis import locs_to_vals
from scipy.interpolate import CubicSpline
import pleiades.eqformatting as eqfrmt
import h5py
# from pleiades.h5py_helper import dic_to_h5grp, h5grp_to_dic

class EQDSK():
    def __init__(self,fname=None,eqdict=None):
        if fname is not None:
            # print('creating eqdsk class from file')
            self.read_eqdsk_file(fname)
        elif eqdict is not None:
            print('creating eqdsk class from dict')
            self.read_eqdsk_dict(eqdict)
        else:
            print('creating empty eqdsk class')

    def create_eqdsk(self,R,Z,psi,rLim,zLim,title):
    #
    # Generate eqdsk class and fill in class variables from pleiades eq. soln.
    #
    # Inputs:
    # R
    # Z
    # psi
    # rLim
    # zLim
    # title
    #
        #generate limiter and vessel surfaces
        print('creating_eqdsk')
        [psi_lim] = locs_to_vals(R,Z,psi,[(rLim,zLim)])
        cs_ = plt.contour(Z,R,psi,levels=[psi_lim])
        limdat = cs_.allsegs[0]
        if np.array(limdat).ndim > 2:
            limdat = limdat[0]
        plt.close()
        
        psi_ves = psi_lim * 1.1
        cs_ = plt.contour(Z,R,psi,levels=[psi_ves])
        vesdat = cs_.allsegs[0]
        if np.array(vesdat).ndim > 2:
            vesdat = vesdat[0]
        plt.close()
        
        self.title = title
        if len(self.title) >= 35:
            self.title = self.title[0:35]
        self.title = self.title.ljust(35)
        self.nr, self.nz = int(len(R[0,:])), int(len(Z[:,0]))
        self.rdim, self.zdim = np.amax(R)-np.amin(R), np.amax(Z)-np.amin(Z)
        self.sibry = psi_lim
        
        #these values are either zero or not well defined for linear geometry
        self.rcenter, self.rleft, self.zmid, self.current  = 0.0, 0.0, 0.0, 0.0 
        self.rmaxis, self.zmaxis, self.simag, self.bcentr = 0.0, 0.0, 0.0, 0.0

        blank = np.zeros(self.nr)
        self.fpol = blank
        self.pres = blank
        self.ffprim = blank
        self.pprime = blank
        self.qpsi = blank

        #2D arrays
        self.psirz = psi
        self.rlim = limdat[:,1]
        self.zlim = limdat[:,0]
        self.rves = vesdat[:,1]
        self.zves = vesdat[:,0]
        self.nlim = len(self.rlim)
        self.nves = len(self.rves)
        print('added soln to eqdsk class')
        
    def read_eqdsk_file(self, filename):
    #
    # Reads eqdsk file and assigns associated class variables
    #
    # Inputs:
    # filename - file name of the eqdsk file to be read
    #    
        eqdsk = open(filename,"r")
        
        #this is identical to the read in from fortran used in the g_eqdsk 
        #code from princeton, see eqdsk documentation
        self.title, self.nr, self.nz = eqfrmt._read_header(eqdsk)

        if len(self.title) >= 35:
            self.title = self.title[0:35]
        self.title = self.title.ljust(35)
        
        self.rdim, self.zdim, self.rcenter, self.rleft, self.zmid = eqfrmt._read2000format(eqdsk)
        self.rmaxis, self.zmaxis, self.simag, self.sibry, self.bcentr = eqfrmt._read2000format(eqdsk)
        self.current, self.simag, dummy, self.rmaxis, dummy  = eqfrmt._read2000format(eqdsk)
        self.zmaxis, dummy, self.sibry, dummy, dummy = eqfrmt._read2000format(eqdsk)
                
        self.fpol = eqfrmt._read2020format(eqdsk,self.nr)
        self.pres = eqfrmt._read2020format(eqdsk,self.nr)
        self.ffprim = eqfrmt._read2020format(eqdsk,self.nr)
        self.pprime = eqfrmt._read2020format(eqdsk,self.nr)
        self.psirz = eqfrmt._read2020format(eqdsk,self.nz,self.nr)
        self.qpsi = eqfrmt._read2020format(eqdsk,self.nr)        

        #note that i use different naming here than the typical documentation
        #i call bbbs values lim and limtr values ves. this is more intuitive naming
        #to me since the bbbs value are the limiting surface and limtr is the vessel
        try:
            self.nlim, self.nves = eqfrmt._read2022format(eqdsk)
            self.rlim, self.zlim = eqfrmt._readmod2020format(eqdsk, self.nlim)
            self.rves, self.zves = eqfrmt._readmod2020format(eqdsk,self.nves)
        except:
            print('Nonstandard limiter/vessel found setting number of limiter points to zero')
            self.nlim, self.nves = 0, 0
        eqdsk.close()

    def read_eqdsk_dict(self, eqDict):
    #
    # Reads dictionary containing eqdsk values and assigns associated class variables
    #
    # Inputs:
    # eqDict - dictionary containing eqdsk values
    # 
        #header values
        self.title = eqDict['title']
        if len(self.title) >= 35:
            self.title = self.title[0:35]
        self.title = self.title.ljust(35)
        self.nr, self.nz = eqDict['nr'], eqDict['nz']
        #scalars
        self.rdim, self.zdim = eqDict['rdim'], eqDict['zdim']
        self.rcenter, self.rleft, self.zmid = eqDict['rcenter'],  eqDict['rleft'], eqDict['zmid']
        self.rmaxis, self.zmaxis,self.current = eqDict['rmaxis'], eqDict['zmaxis'], eqDict['current']
        self.simag, self.sibry, self.bcentr = eqDict['simag'], eqDict['sibry'], eqDict['bcentr'] 
        #1d arrays
        self.fpol = eqDict['fpol']
        self.pres = eqDict['pres']
        self.ffprim = eqDict['ffprim']
        self.pprime = eqDict['pprime']
        self.qpsi = eqDict['qpsi']
        #2d arrays
        self.psirz = eqDict['psirz']
        #limiting surface and vessel
        self.nlim = eqDict['nlim']
        self.nves = eqDict['nves']
        self.rlim = eqDict['rlim']
        self.zlim = eqDict['zlim']
        self.rves = eqDict['rves']
        self.zves = eqDict['zves']
        
    def write_eqdskdata(self,filename):
    #
    # Writes standard format eqdsk file
    #
    # Inputs:
    # filename - filename to be written to
    # 
       eqdsk = open(filename,'w')
       xdum = 0.0
      
       eqfrmt._writeheader(eqdsk,self.title,self.nr,self.nz)
      
       temp = self.rdim,self.zdim,self.rcenter,self.rleft,self.zmid
       eqfrmt._write2000format(eqdsk,temp)
       temp = self.rmaxis,self.zmaxis,self.simag,self.sibry,self.bcentr
       eqfrmt._write2000format(eqdsk,temp)
       temp = self.current,self.simag,xdum,self.rmaxis,xdum
       eqfrmt._write2000format(eqdsk,temp)
       temp = self.zmaxis,xdum,self.sibry,xdum,xdum
       eqfrmt._write2000format(eqdsk,temp)
       
       eqfrmt._write2020format(eqdsk,self.fpol)
       eqfrmt._write2020format(eqdsk,self.pres)
       eqfrmt._write2020format(eqdsk,self.ffprim)
       eqfrmt._write2020format(eqdsk,self.pprime)
       eqfrmt._write2020format(eqdsk,self.psirz.flatten())
       eqfrmt._write2020format(eqdsk,self.qpsi)
       
       eqdsk.write('{:5d}{:5d}\n'.format(self.nlim,self.nves))
       eqfrmt._writemod2020format(eqdsk,self.rlim,self.zlim)
       eqfrmt._writemod2020format(eqdsk,self.rves,self.zves)
       eqdsk.close()

    def write_to_dict(self):
    #
    # Writes contents of class to a dictionary
    #
    # Returns:
    # eqDict - dictionary containing eqdsk class variables
    #
        excluded_keys = []
        eqDict = {key:value for key, value in self.__dict__.items() \
                  if not key.startswith('__') and not callable(key)}

        return eqDict
    
class H5(): 
    def __init__(self,fname=None):
        self.eqFull=False
        self.cqlFull=False
        if fname is not None:
            print('creating H5 class from file')
            self.fname = fname
            self.fill_class_from_file()
        else:
            print('creating H5 class')

    def fill_class_from_file(self):
        pass
            
    def fill_eq(self,R,Z,B,Br,Bz,eqdsk):
        self.R = R #could pull from eqdsk object but pain
        self.Z = Z
        self.psirz = eqdsk.psirz
        self.B = B
        self.B_Z = Bz
        self.B_R = Br
        self.limZ = eqdsk.zlim
        self.limR = eqdsk.rlim
        self.vesZ = eqdsk.zves
        self.vesR = eqdsk.rves
        self.eqFull = True
        
    def fill_cql(self,anisoEq):
        self.RHalf = anisoEq.r_half
        self.ZHalf = anisoEq.z_half
        self.BHalf = anisoEq.B
        self.BrHalf = anisoEq.BrRatio_interp((anisoEq.r_half,anisoEq.z_half))
        self.BzHalf = anisoEq.BzRatio_interp((anisoEq.r_half,anisoEq.z_half))
        self.JprpHalf = anisoEq.J_half
        self.psiCql = anisoEq.psi
        self.p_prp = anisoEq.p_prp
        self.p_par = anisoEq.p_par
        self.lHalf = anisoEq.l_half
        self.cqlFull = True
        
    def create_file(self):
    #
    # Generate H5 and fill in class variables from pleiades eq. soln
    # quick and dirty need this off my plate
    #
        print('creating output HDF5')
        outH5 = h5py.File('pleiades.h5','w')
        if self.eqFull:
            eqGrp = outH5.create_group("eqdsk")
            eqGrp.create_dataset('R',data=self.R)
            eqGrp.create_dataset('Z',data=self.Z)
            eqGrp.create_dataset('psi',data=self.psirz)
            eqGrp.create_dataset('B',data=self.B)
            eqGrp.create_dataset('B_Z',data=self.B_Z)
            eqGrp.create_dataset('B_R',data=self.B_R)
            eqGrp.create_dataset('limR',data=self.limR)
            eqGrp.create_dataset('limZ',data=self.limZ)
            eqGrp.create_dataset('vesR',data=self.vesR)
            eqGrp.create_dataset('vesZ',data=self.vesZ)
        if self.cqlFull:
            cqlGrp = outH5.create_group("aniso")
            cqlGrp.create_dataset('psiCql',data=self.psiCql)
            cqlGrp.create_dataset('RCql',data=self.RHalf)
            cqlGrp.create_dataset('ZCql',data=self.ZHalf)
            cqlGrp.create_dataset('lCql',data=self.lHalf)
            cqlGrp.create_dataset('BCql',data=self.BHalf)
            cqlGrp.create_dataset('B_rCql',data=self.BrHalf)
            cqlGrp.create_dataset('B_zCql',data=self.BzHalf)
            cqlGrp.create_dataset('p_prp',data=self.p_prp)
            cqlGrp.create_dataset('p_par',data=self.p_par)
            cqlGrp.create_dataset('J_prpCql',data=self.JprpHalf)
            
        outH5.close()

   
