# A class module to compute gradients in 2D spatial data

from __future__ import print_function
import astropy.units as u

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
    def compute(filename, image=True, columns=[],
                outputdir='./', outputformat='ascii',
                wcs=None, distance=None, compute_over_map=True,
                blocksize=5, minnpix=None, pinit=[]):

        """
        compute gradient

        parameters
        ----------
        filename : string
            filename (including path) to data file
        image : bool
            indicate whether the data is in image format (2D) array or tabular
            default=True
        columns : list
            if data is passed in tabular format, you should indicate which
            columns correspond to [x,y,data]
        outputdir : string
            output directory for output files
        outputformat : string
            output format for the gradient computation default='ascii' - a
            tabular output of measurements, though 'fits' or 'both' are accepted
        wcs :

        distance : float with units
            provide the distance as with corresponding units (use astropy.units)
            this is used to convert the gradients into physical units
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

        """

        # TODO: need to put some assertions in here

        self=gradients()
        self.image=image
        self.columns=columns
        self.outputdir=outputdir
        self.outputformat=outputformat
        self.wcs=wcs
        self.distance=distance
        self.compute_over_map=compute_over_map
        self.blocksize=blocksize
        self.minnpix=minnpix
        self.pinit=pinit

        self.data=unpack_data(self)


    def unpack_data(self):
        """
        Here we are going to unpack the data in a format that we can use for
        gradient computation and one that is independent of input file format
        i.e. an image vs. an ascii table
        """
        from astropy.io import fits, ascii

        if self.image:
            d=fits.open(self.filename)[0].data
            _xx,_yy = np.meshgrid(np.arange(np.shape(d)[1]), np.arange(np.shape(d)[0]))
            ypos,xpos=np.where(~np.isnan(d))
            x,y,data=_xx[ypos,xpos], _yy[ypos,xpos], d[ypos,xpos]
        else:
            d=ascii.read(self.filename)


        return data
