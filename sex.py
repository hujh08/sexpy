#!/usr/bin/env python3

# class to handle sextractor's result
#   mainly output catalog and segmentation, a check image

import os

from math import floor, ceil
import numpy as np

from astropy.io import fits
from astropy import wcs

import warnings

class Sex:
    def __init__(self, *args, **kwargs):
        '''
            initiate via
                sex configure file
                or some sex result files
        '''
        if 'config' in kwargs or len(args)<=1:
            self._parse_config(*args, **kwargs)
        else:
            self._init(*args, **kwargs)

    def _init(self, catalog, seg, type='ASCII_HEAD',
                       bkg=None, bkgrms=None):
        '''
            initiate with some sex result files
        '''
        self._load_catalog(catalog, type)
        self._load_seg(seg)

        if bkg!=None:
            self.bkg=fits.getdata(bkg)

        if bkgrms!=None:
            self.bkgrms=fits.getdata(bkgrms)

    def _parse_config(self, config='default.sex'):
        '''
            load sex configuration file
                instead of specifying parameters
        '''
        params=self._load_config(config)
        self.params=params

        cat_name,=params['CATALOG_NAME']
        cat_type,=params['CATALOG_TYPE']

        # check images
        check_images=dict(zip(params['CHECKIMAGE_TYPE'],
                              params['CHECKIMAGE_NAME']))
        seg=check_images['SEGMENTATION']

        opts={}
        if 'BACKGROUND' in check_images:
            opts['bkg']=check_images['BACKGROUND']
        if 'BACKGROUND_RMS' in check_images:
            opts['bkgrms']=check_images['BACKGROUND_RMS']

        self._init(cat_name, seg, type=cat_type, **opts)

    def _load_config(self, config):
        '''
            load sex configure file

            return a dict {key: value}
        '''
        # handle sex result files
        dir_sex=os.path.dirname(config)
        key_sex_files=set(['CATALOG_NAME', 'CHECKIMAGE_NAME'])

        params={}
        with open(config) as f:
            for l in f:
                l=l.strip()
                if not l or l[0]=='#':
                    continue
                key, vals=l.split('#')[0].split(maxsplit=1)
                
                vals=[s.strip() for s in vals.split(',')]
                if key in key_sex_files:
                    vals=[os.path.join(dir_sex, t) for t in vals]

                params[key]=vals
        return params

    # load result of sextractor
    def _load_catalog(self, catalog, type):
        '''
            information from catalog,
                stored as a numpy record type
        '''
        func=getattr(self, '_load_catalog_%s' % type.upper())
        func(catalog)

    def _handle_line(self, line, fmts, vessels):
        '''
            auxiliary function to load ascii_head catalog
        '''
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

    def indseg_from_pix(self, x, y):
        '''
            index of segment for given pixel coordinates `x`, `y`
        '''
        x, y=floor(x), floor(y)
        seg=self.segs[y-1, x-1]
        if seg==0:
            raise Exception('no segment found')
        return seg

    def indseg_from_radec(self, ra, dec):
        '''
            return integer index of segment for giving ra dec
                started from 1
        '''
        return self.indseg_from_pix(*self.radec2pix(ra, dec))

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

    def parameters_seg(self, seg):
        '''
            parameters in catalog for a given seg
        '''
        return self.catas[seg-1]

    def parameters(self, ra, dec):
        '''
            parameters in catalog for a giving ra dec
        '''
        seg=self.indseg(ra, dec)
        return self.parameters_seg(seg)

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

    def gfpars_seg(self, seg):
        '''
            all the parameters in output catalog
                which will be used in galfit directly
                for a given seg
        '''
        pars=self.parameters_seg(seg)
        x0=pars['X_IMAGE']
        y0=pars['Y_IMAGE']
        mag=pars['MAG_AUTO']
        re=pars['FLUX_RADIUS']
        ba=1-pars['ELLIPTICITY']

        pa=pars['THETA_IMAGE']+90
        if pa>=90:
            pa-=180

        return x0, y0, mag, re, ba, pa

    def gfpars(self, ra, dec):
        '''
            same as above, but giving ra, dec, not seg
        '''
        seg=self.indseg(ra, dec)

        return self.gfpars_seg(seg)

    # background
    def _bkg_region(self, fits, region, exclude=True):
        '''
            average background or bkg rms inside the given region

            Parameters
            ----------
            exclude: boolean
                whether exclude objects before yielding bkg
        '''
        xmin, xmax, ymin, ymax=region

        bkg=fits[(ymin-1):ymax, (xmin-1):xmax]

        if exclude:
            segs=self.segs[(ymin-1):ymax, (xmin-1):xmax]
            bkg=bkg[segs==0]

        return bkg.mean()

    def bkg_mean_region(self, *args, **kwargs):
        return self._bkg_region(self.bkg, *args, **kwargs)

    def bkg_region(self, *args, **kwargs):
        return self.bkg_mean_region(*args, **kwargs)

    def bkg_rms_region(self, *args, **kwargs):
        return self._bkg_region(self.bkgrms, *args, **kwargs)

    # fit region for galfit
    def fitregion(self, seg, margin=20,
                             center_match=True,
                             clip50=True):
        '''
            use the seg region to determine fit region
                with expanding out by `margin` pixels

            optional Parameters:
            ----------
            center_match: boolean
                whether centers of target seg and fit region match

            clip50: boolean
                whether clip the semi-width to times of 50
        '''
        xmin, xmax, ymin, ymax=self.region_seg(seg)

        xmin-=margin
        xmax+=margin
        ymin-=margin
        ymax+=margin

        if center_match:
            pars=self.parameters_seg(seg)
            x0=pars['X_IMAGE']
            y0=pars['Y_IMAGE']

            semiwx=max(x0-xmin, xmax-x0)
            semiwy=max(y0-ymin, ymax-y0)

            if clip50:
                semiwx=ceil(semiwx/50)*50
                semiwy=ceil(semiwy/50)*50
            else:
                semiwx=ceil(semiwx)
                semiwy=ceil(semiwy)

            x0=int(x0)
            y0=int(y0)

            xmin=x0-semiwx
            xmax=x0+semiwx
            ymin=y0-semiwy
            ymax=y0+semiwy

        # clip the exceed region
        ny, nx=self.segs.shape
        if xmin<1:
            xmin=1
        if xmax>nx:
            xmax=nx
        if ymin<1:
            ymin=1
        if ymax>ny:
            ymax=ny

        return xmin, xmax, ymin, ymax

    # create mask
    def _mask_array(self, seg):
        '''
            all the seg except specified `seg` are excluded in galfit

            return 2d array in which
                bad pixels are marked with value > 0
        '''
        mask=self.segs.copy()
        mask[mask==seg]=0
        return mask

    def _mask_xy(self, seg, region=None):
        '''
            return two list of x,y coordinates of seg pixels
                within `region`
                    if given, with format [xmin, xmax, ymin, ymax]
                        which start from 1
                    if not, means whole image

            mask gives the name of mask file
        '''
        ycoords, xcoords=np.indices(self.segs.shape)+1
        mask=self._mask_array(seg)

        badp=mask>0

        ycoords=ycoords[badp]
        xcoords=xcoords[badp]

        if region!=None:
            xmin, xmax, ymin, ymax=region
            inreg=(xmin-1<=xcoords)*(xcoords<=xmax-1)*\
                  (ymin-1<=ycoords)*(ycoords<=ymax-1)

            xcoords=xcoords[inreg]
            ycoords=ycoords[inreg]

        return np.c_[xcoords, ycoords]

    def mask_text(self, seg, mask, region=None):
        '''
            save x,y of bad pixels in an ASCII text file
        '''
        with open(mask, 'w') as f:
            for x, y in self._mask_xy(seg, region):
                f.write('%i %i\n' % (x, y))

    def mask_fits(self, seg, mask, overwrite=True):
        '''
            save x,y of bad pixels in a FITS file
        '''
        fits.writeto(mask, self._mask_array(seg),
                     overwrite=overwrite)

    # detect weird object
    def is_longslit(self, seg, max_reg=0.8):
        '''
            target object is a long slit, maybe a cosmic ray
        '''
        ny, nx=np.array(self.segs.shape)*max_reg

        xmin, xmax, ymin, ymax=self.region_seg(seg)

        if (xmax-xmin+1)>=nx or (ymax-ymin+1)>=ny:
            return 'yes'

        return 'no'
