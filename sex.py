#!/usr/bin/env python3

# class to handle sextractor's result
#   mainly output catalog and segmentation, a check image

from math import floor
import numpy as np

from astropy.io import fits
from astropy import wcs

import warnings

class Sex:
    def __init__(self, catalog, seg, type='ASCII_HEAD'):
        self._load_catalog(catalog, type)
        self._load_seg(seg)

    # load result of sextractor
    def _load_catalog(self, catalog, type):
        '''
        information from catalog,
            stored as a numpy record type
        '''
        func=getattr(self, '_load_catalog_%s' % type.upper())
        func(catalog)

    def _handle_line(self, line, fmts, vessels):
        # auxiliary function to load ascii_head catalog
        for s, func, vess in zip(line.split(), fmts, vessels):
            vess.append(func(s))

    def _load_catalog_ASCII_HEAD(self, catalog):
        with open(catalog) as f:
            columns=[]
            datas=[]
            fmts=[]

            for line in f:
                if line[0]=='#':
                    name=line.split()[2]
                    columns.append(name)

                    datas.append([])

                    if name in ['NUMBER', 'FLAGS']:
                        fmts.append(int)
                    else:
                        fmts.append(float)
                else:
                    break

            if line and line[0]!='#':
                self._handle_line(line, fmts, datas)
                for line in f:
                    self._handle_line(line, fmts, datas)

            dtype=list(zip(columns, fmts))
            self.catas=np.rec.fromarrays(datas, dtype=dtype)

    def _load_catalog_FITS_LDAC(self, catalog):
        datas=fits.open(catalog)[-1].data

        names=datas.columns.names
        cols=[datas[col] for col in names]
        self.catas=np.rec.fromarrays(cols, names=names)

    def _load_seg(self, segfits):
        '''
        information from segmentation image:
            1, index of object in catalog for each pixel
            2, world coordination
        '''
        hdu=fits.open(segfits)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.w=wcs.WCS(hdu.header)

        # indices in output catalog, started from 1
        self.segs=hdu.data

    # handle query
    def radec2pix(self, ra, dec):
        '''
        convert world coordinate to pixel coordinate
        '''
        return self.w.wcs_world2pix(ra, dec, True)

    def indpix(self, ra, dec):
        '''
        return integer index of pixel for giving ra dec
            with 1,1 for the left-bottom corner
        '''
        x, y=self.radec2pix(ra, dec)
        return floor(x), floor(y)

    def indseg(self, ra, dec):
        '''
        return integer index of segment for giving ra dec
            started from 1
        '''
        x, y=self.indpix(ra, dec)
        return self.segs[y-1, x-1]

    def region_seg(self, seg):
        '''
        region for a giving segment
            represented by (xmin, xmax, ymin, ymax)
                in which index starts from 1

        seg is integer index of segment, starting from 1
            as in seg image
        '''
        ycoords, xcoords=np.indices(self.segs.shape)
        pbool=(self.segs==seg)

        ytarget=ycoords[pbool]
        xtarget=xcoords[pbool]
        xmin=xtarget.min()+1
        xmax=xtarget.max()+1
        ymin=ytarget.min()+1
        ymax=ytarget.max()+1

        return xmin, xmax, ymin, ymax

    def segregion(self, ra, dec):
        '''
        region for a giving ra dec

        just combine indseg and region_seg method
        '''
        seg=self.indseg(ra, dec)

        return self.region_seg(seg)

    def parameters(self, ra, dec):
        '''
        parameters in catalog for a giving ra dec
        '''
        seg=self.indseg(ra, dec)
        return self.catas[seg-1]

    ## some frequently queried parameters
    def xcent(self, ra, dec):
        return self.parameters(ra, dec)['X_IMAGE']

    def ycent(self, ra, dec):
        return self.parameters(ra, dec)['Y_IMAGE']

    def ellip(self, ra, dec):
        return self.parameters(ra, dec)['ELLIPTICITY']

    def pa(self, ra, dec):
        '''
        galfit standard position angle
            which means 0 for up
        '''
        pa=self.parameters(ra, dec)['THETA_IMAGE']+90
        if pa>=90:
            pa-=180
        return pa

    def re(self, ra, dec):
        '''
        effective radius, inside which contains 50% flux
            this 50% fraction is specified 
                by PHOT_FLUXFRAC in configuration file
        '''
        return self.parameters(ra, dec)['FLUX_RADIUS']

    def mag(self, ra, dec):
        '''
        magnitude with zero point=20
            which is also specified in configuration
                by MAG_ZEROPOINT key word
        '''
        return self.parameters(ra, dec)['MAG_AUTO']

    def gfpars(self, ra, dec):
        '''
        all the parameters in output catalog
            which will be used in galfit
        '''
        pars=self.parameters(ra, dec)
        x0=pars['X_IMAGE']
        y0=pars['Y_IMAGE']
        mag=pars['MAG_AUTO']
        re=pars['FLUX_RADIUS']
        ba=pars['ELLIPTICITY']

        pa=pars['THETA_IMAGE']+90
        if pa>=90:
            pa-=180

        return x0, y0, mag, re, ba, pa