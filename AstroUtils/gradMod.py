# A class module to compute gradients in 2D spatial data

from __future__ import print_function
import astropy.units as u
import numpy as np

class gradients(object):
    """
    gradients
    ---------

    Used to compute gradients in 2D spatial data. Can be used to compute
    gradients across an astronomical image, or in smaller regions of a map.
    Can be used with 2D images or with a table housing data extracted from a
    regular grid.
    """
    def __init__(self):

        self.filename=None
        self.data=None
        self.image=None
        self.wcs=None
        self.distance=None
        self.outputdir=None
        self.outputformat=None

        self.compute_over_map=None
        self.blocksize=None
        self.minnpix=None

        self.pinit=None

        self.grad, self.grad_err=None,None
        self.angle, self.angle_err=None,None
        self.gradx, self.gradx_err=None,None
        self.grady, self.grady_err=None,None

        def static_warning(self, *args, **kwargs):
            err = "Invalid use of static method. Try grad=gradients.compute(filename)"
            raise AttributeError(err)

        self.compute = static_warning

    @staticmethod
    def compute(filename, image=True, errimage=None, columns=[],
                outputdir='./', outputformat='ascii',
                physical_units=True,
                wcs=None, distance=None,
                compute_over_map=True,
                shape=None, blocksize=5, minnpix=None, pinit=None,
                report_fit=False):

        """
        compute gradient

        parameters
        ----------
        filename : str or `pathlib.Path`
            filename (including path) to data file
        image : bool
            indicate whether the data is in image format (2D) array or tabular
            default=True. Must be in FITS format
        errimage : str or `pathlib.Path`
            filename (including path) to error image if one is available
        columns : list
            if data is passed in tabular format, you should indicate which
            columns correspond to [x,y,data]. If passing the data in tabular
            format, please pass the x,y coordinates in pixel coordinates. Note
            that you can also pass a column of uncertainties. It will be assumed
            that this is index=3 in the list
        outputdir : str
            output directory for output files
        outputformat : str
            output format for the gradient computation default='ascii' - a
            tabular output of measurements, though 'fits' or 'both' are accepted
        physical_units : bool
            gradient computation is performed on pixel units toggle this and use
            keywords below to convert gradients to physical units
        dataunit : str
            pass the data units strings e.g. 'km/s' default='km/s'
        wcs : astropy.wcs
            world coordinate system for conversion to physical units
        distance : float with units
            provide the distance in parsecs this is used to convert the gradients
            into physical units
        compute_over_map : bool
            compute the gradient over entire data set or over segments of data
            default=True which means the computation will be performed over the
            whole map and a single value returned
        blocksize : int
            used if compute_over_map=False - computes gradient in blocksize
            sized boxes over the map - used for local measurements of gradients
            output will be a tabulated list of local gradient measurements
        minnpix : int
            minimum number of pixels in block for gradient to be computed - if
            npix < minnpix within a block of blocksize^2 then no grad will be
            computed for that block
        pinit : list
            initial values for least squares minimisation
        report_fit : bool
            prints the lmfit result to terminal


        """

        # TODO: need to put some assertions in here

        self=gradients()
        self.filename=filename
        self.image=image
        self.errimage=errimage
        self.columns=columns
        self.outputdir=outputdir
        self.outputformat=outputformat
        self.physical_units=physical_units
        self.wcs=wcs
        self.distance=distance
        self.compute_over_map=compute_over_map
        self.shape=shape
        self.blocksize=blocksize
        self.minnpix=minnpix
        self.pinit=pinit
        self.report_fit=report_fit

        # set this automatically if not set by the user
        if self.minnpix is None:
            if self.compute_over_map:
                self.minnpix=0
            else:
                self.minnpix=int((self.blocksize**2)*0.6)

        # unpack the data for gradient computation
        self.data=self.unpack_data()
        # get number of measurements for grad comp
        nummeas=1 if self.compute_over_map else np.shape(self.data)[1]
        self.gradtab=self.calculate_gradients(nummeas=nummeas)

        id=[j for j in range(len(self.gradtab['x'])) if ((self.gradtab['x'][j]==489.0) & (self.gradtab['y'][j]==330.0)) ]

        # at this point we have a table of gradients in data unit per pixel units
        # now we can convert this to physical units
        if self.physical_units:
            conversion_factor=self.compute_conversion_factor()
            self.gradtab=self.modify_gradtab(conversion_factor)



    def unpack_data(self):
        """
        Here we are going to unpack the data in a format that we can use for
        gradient computation and one that is independent of input file format
        i.e. an image vs. an ascii table
        """
        from astropy.io import fits, ascii

        if self.image:
            d=fits.open(self.filename)[0].data
            if self.errimage is not None:
                e=fits.open(self.filename)[0].data
            else:
                e=np.copy(d)*0.0
            _xx,_yy = np.meshgrid(np.arange(np.shape(d)[1]), np.arange(np.shape(d)[0]))
            ypos,xpos=np.where(~np.isnan(d))
            x,y,data,error=_xx[ypos,xpos], _yy[ypos,xpos], d[ypos,xpos], e[ypos,xpos]
        else:
            d=ascii.read(self.filename)
            headings=list(d.columns)
            if 'ncomps' in headings:
                d=d[d['ncomps']>0]
            if np.size(self.columns)==0:
                raise AttributeError("Please indicate columns to be read in format columns=[x_index,y_index,data_index, error_index]")
            elif np.size(self.columns)==3:
                x,y,data,error=d[self.columns[0]][:].data, d[self.columns[1]][:].data, d[self.columns[2]][:].data, np.copy(d[self.columns[2]][:].data)*0.0
            elif np.size(self.columns)==4:
                x,y,data,error=d[self.columns[0]][:].data, d[self.columns[1]][:].data, d[self.columns[2]][:].data, d[self.columns[3]][:].data
            else:
                raise AttributeError("Number of columns not recognised, columns should have length 3 or 4")
        return np.asarray([x,y,data,error])

    def calculate_gradients(self, nummeas=1):
        """
        module for gradient computation

        parameters
        ----------
        nummeas : int
            number of times we are going to perform the calculation
        """
        from astropy.table import Table
        from astropy.table import Column, Row
        from tqdm import tqdm

        # first thing - lets create an empty table to house the gradient
        # measurements
        gradtab_headings=['x', 'y', 'grad', 'err grad', 'grad_x', 'err_grad_x', 'grad_y', 'err_grad_y', 'theta', 'err theta']
        gradtab=Table(meta={'name': 'params'}, names=gradtab_headings)

        if self.compute_over_map:
            if np.sum(self.data[3,:])==0.0:
                model, errors, result=self.fit(self.data[0,:],self.data[1,:],self.data[2,:],
                                               pinit=self.pinit, report_fit=self.report_fit)
            else:
                model, errors, result=self.fit(self.data[0,:],self.data[1,:],self.data[2,:],
                                               err=self.data[3,:],
                                               pinit=self.pinit, report_fit=self.report_fit)
            if None in errors:
                errors=np.array([0.0,0.0,0.0])

            cols=self.return_cols(0,0,model,errors)

            gradtab.add_row(cols)

        else:
            if self.shape is None:
                err="If passing the data in tabular form, please pass the shape of the original array "
                err+="using shape=[yy,xx]"
                raise AttributeError(err)

            for i in tqdm(range(nummeas)):
                # find the neighbouring pixels
                keys=self.get_neighbours(self.data[0,i], self.data[1,i], self.shape, self.blocksize)
                # unravel and get their pix coords
                neighboursy, neighboursx = np.unravel_index(keys, self.shape, order='C')
                # find where these are in our array
                idn=[idneigh[0][0] if np.size(idneigh)!=0 else np.nan
                     for idneigh in [np.where((self.data[0,:]==neighboursx[k]) & (self.data[1,:]==neighboursy[k]))
                     for k in range(len(neighboursy))]]
                # get rid of any nans
                idn=[id for id in idn if np.isfinite(id)]

                # now if the size of the neighbouring pixel list is greater than
                # our minnpix value, compute the local gradient
                if np.size(idn)>=self.minnpix:
                    if np.sum(self.data[3,:])==0.0:
                        model, errors, result=self.fit(self.data[0,idn],self.data[1,idn],self.data[2,idn],
                                                       pinit=self.pinit, report_fit=False)
                    else:
                        model, errors, result=self.fit(self.data[0,idn],self.data[1,idn],self.data[2,idn],
                                                       err=self.data[3,idn],
                                                       pinit=self.pinit, report_fit=False)
                    if None in errors:
                        errors=np.array([0.0,0.0,0.0])

                    cols=self.return_cols(self.data[0,i],self.data[1,i],model,errors)

                    gradtab.add_row(cols)

        return gradtab

    def return_cols(self, x, y, model, errors):
        """
        Returns a list of gradient information in a format that can be added to
        the table
        """
        import uncertainties.unumpy as unp

        grad_x_pix=unp.uarray(model[0],errors[0])
        grad_y_pix=unp.uarray(model[1],errors[1])
        grad_pix=unp.sqrt(grad_x_pix**2 + grad_y_pix**2)
        orientation = np.rad2deg(unp.nominal_values(unp.arctan2(grad_x_pix, grad_y_pix)))
        orientation_error = np.rad2deg(unp.nominal_values(unp.std_devs(unp.arctan2(grad_x_pix, grad_y_pix))))

        cols = [x,y,unp.nominal_values(grad_pix),unp.std_devs(grad_pix),
                unp.nominal_values(grad_x_pix),unp.std_devs(grad_x_pix),
                unp.nominal_values(grad_y_pix),unp.std_devs(grad_y_pix),
                orientation, orientation_error]

        return cols

    def get_neighbours(self, x, y, shape, blocksize):
        """
        Returns a list of flattened indices for a given spectrum and its neighbours

        Parameters
        ----------
        x : int
            x position of the reference pixel
        y : int
            y position of the reference pixel
        """

        neighboursx=np.arange(x-(blocksize-1)/2,(x+(blocksize-1)/2)+1,dtype='int' )
        neighboursx=[x if (x>=0) & (x<=shape[1]-1) else np.nan for x in neighboursx ]
        neighboursy=np.arange(y-(blocksize-1)/2,(y+(blocksize-1)/2)+1,dtype='int' )
        neighboursy=[y if (y>=0) & (y<=shape[0]-1) else np.nan for y in neighboursy ]
        keys=[np.ravel_multi_index([y,x], shape)  if np.all(np.isfinite(np.asarray([y,x]))) else np.nan for y in neighboursy for x in neighboursx]

        return keys

    def fit(self, x, y, z, err=None, pinit=None, method='leastsq', report_fit=False):
        """
        Fits a first-degree bivariate polynomial to 2D data

        parameters
        ----------
        x : ndarray
            array of x values
        y : ndarray
            array of y values
        z : ndarray
            data to be fit
        err : ndarray (optional)
            uncertainties on z data
        pinit : ndarray (optional)
            initial guesses for fitting. Format = [mx, my, c]
        method : string (optional)
            method used for the minimisation (default = leastsq)

        """
        import lmfit

        if pinit is None:
            pars=lmfit.Parameters()
            pars.add('mx', value=1.0)
            pars.add('my', value=1.0)
            pars.add('c', value=1.0)
        else:
            pars=lmfit.Parameters()
            pars.add('mx', value=pinit[0])
            pars.add('my', value=pinit[1])
            pars.add('c', value=pinit[2])

        fitter = lmfit.Minimizer(self.residual, pars,
                                 fcn_args=(x,y),
                                 fcn_kws={'data':z, 'err':err},
                                 nan_policy='propagate')

        result = fitter.minimize(method=method)

        if report_fit:
            lmfit.report_fit(result)

        popt = np.array([result.params['mx'].value,
                         result.params['my'].value,
                         result.params['c'].value])
        perr = np.array([result.params['mx'].stderr,
                         result.params['my'].stderr,
                         result.params['c'].stderr])

        return popt, perr, result

    def residual(self, pars, x, y, data=None, err=None):
        """
        Minmizer for lmfit for fitting a 2-D plane to data

        parameters
        ----------
        pars : lmfit.Parameters()

        x : ndarray
            array of x positions
        y : ndarray
            array of y positions
        data : ndarray
            2-D image containing the data
        err : ndarray
            uncertainties on the data

        """
        parvals = pars.valuesdict()
        mx = parvals['mx']
        my = parvals['my']
        c = parvals['c']
        model = self.polynomial(x, y, mx, my, c)

        if data is None:
            min = np.array([model])
            return min
        if err is None:
            min = np.array([model - data])
            return min
        min = np.array([(model-data) / err])

        return min

    def polynomial(self, x, y, mx, my, c):
        """
        A polynomial function for fitting 2D data

        Parameters
        ----------
        x : ndarray
            array of x values
        y : ndarray
            array of y values
        mx : float
            gradient in x
        my : float
            gradient in y
        c : float
            offset

        """
        return mx*x + my*y + c

    def compute_conversion_factor(self):
        """
        Here we are going to compute the conversion factor to convert from pixel
        units to world coordinates
        """
        from astropy import wcs
        pixelsize_deg=wcs.utils.proj_plane_pixel_scales(self.wcs)[0]
        pixelsize_parsec=(pixelsize_deg/np.rad2deg(1))*self.distance

        return 1./pixelsize_parsec

    def modify_gradtab(self,conversion_factor):
        """
        modify table by converting gradients to world coordinates

        parameters
        ----------
        conversion_factor : float
            conversion factor to go from pixel coords to world coords
        """
        from astropy.table import Column, Row
        columnids=[2,3,4,5,6,7]
        gradtab_headings=['x', 'y', 'grad', 'err grad', 'grad_x', 'err_grad_x', 'grad_y', 'err_grad_y', 'theta', 'err theta']
        for j in range(len(columnids)):
            self.gradtab[gradtab_headings[columnids[j]]]=self.gradtab[gradtab_headings[columnids[j]]]*conversion_factor

        # in pixel coords x is +ve but the reverse is true in world coords.
        # invert the sign
        self.gradtab['grad_x']=self.gradtab['grad_x']*-1.0
        self.gradtab['theta']=self.gradtab['theta']*-1.0
        # create world coords columns
        worldx, worldy= self.wcs.all_pix2world(self.gradtab['x'],self.gradtab['y'],1)
        worldxcol,worldycol = Column(worldx, 'worldx'), Column(worldy, 'worldy')
        # lets add the world coords to the table
        self.gradtab.add_column(worldxcol, index=2)
        self.gradtab.add_column(worldycol, index=3)

        return self.gradtab
