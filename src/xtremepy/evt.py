try:
    from ulab import numpy
except ImportError:
    import numpy

__all__ = ['ExtremeValues']


class ExtremeValues:

    def __init__(self):
        self._wscale = None
        self._wshape = None
        self._r2 = None

    @property
    def rsquared(self) -> float:
        """The coefficient of determination for fitted parameter"""
        return self._r2

    @property
    def wscale(self) -> float:
        "Scale parameter of Weibull"
        return self._wscale

    @property
    def wshape(self) -> float:
        "Shape parameter of Weibull"
        return self._wshape

    @wscale.setter
    def wscale(self, wsc: float):
        "Set the Scale parameter of Weibull"
        self._wscale = wsc

    @wshape.setter
    def wshape(self, wsh: float):
        "Set the Shape parameter of Weibull"
        self._wshape = wsh

    def reset(self):
        "Reset Weibull Parameter"
        self._wscale = None
        self._wshape = None
        self._r2 = None

    def weibull(self, x: float) -> float:
        """The Weibull distribuction function

        Args:
            x (float): input value

        Returns:
            float: Weibull value for input x.
        """
        if x < 0:
            return 0

        tmp = (self.wshape / self.wscale)
        tmp *= ((x / self.wscale) ** (self.wshape - 1))
        tmp *= numpy.exp(-(x / self.wscale) ** self.wshape)
        return tmp

    def wblcfd(self, x: float) -> float:
        """The Weibull cumulative distribuction function

        Args:
            x (float): input value

        Returns:
            float: CFD of Weibull value for input x.
        """
        return 1.0 - numpy.exp(-(x/self.wscale) ** self.wshape)

    def fit_weibull(self, failures: numpy.ndarray) -> None:
        """The Fit Weibull from array of failures
        Considering the Weibull CFD being:
        F(x) = 1 - exp(-(x/scale) ** shape)

        Is easy to obtain
        y = shape * xl + a

        where:
        y = ln(-ln(1-F(x)))
        xl = ln(x)
        a = - shape * ln(scale)

        From the Least Square Fitting for straight line:

        shape = sum(y*(x-xm)) / (sum(x*(x-xm))

        a = ym-xm * shape

        where xm is the mean of x and ym is the mean of y.
        Args:
            x (ndarray): failures array value
        """
        x = numpy.log(numpy.sort(failures))
        y = (numpy.arange(1, x.size + 1) - 0.5) / x.size
        y[:] = numpy.log(-numpy.log(1-y))
        xm = numpy.mean(x)
        xt = x - xm
        b1 = numpy.sum(y * xt)
        b2 = numpy.sum(x * xt)
        beta = b1 / b2

        a = numpy.mean(y) - xm * beta

        alpha = numpy.exp(-a / beta)

        self.wscale = alpha
        self.wshape = beta

        # Calculating R Squared for
        # y = shape * x - shape * ln(scale)

        def yp(x): return self.wshape * x - \
            self.wshape * numpy.log(self.wscale)
        ym = numpy.mean(y)
        ssres = numpy.sum((y - yp(x)) ** 2.0)
        sstot = numpy.sum((y-ym)**2.0)
        self._r2 = 1.0 - ssres / sstot
