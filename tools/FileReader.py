
class Event(object):
    def __init__(self):
        self.primary = None
        self.energy = None
        self.zenith = None
        self.azimuth = None

        self.xmax = None
        self.nEmAtXmax = None
        self.nEmObslev = None
        
        self.Rcorsika = None
        self.Lcorsika = None
        self.Rfit = None
        self.Lfit = None
        self.sigmaRfit = None
        self.sigmaLfit = None

        self.Xmaxfit = None
        self.sigmaXmaxfit = None

        self.RfitAndringa = None
        self.LfitAndringa = None
        self.sigmaRfitAndringa = None
        self.sigmaLfitAndringa = None

        self.XmaxfitAndringa = None
        self.sigmaXmaxfitAndringa = None
