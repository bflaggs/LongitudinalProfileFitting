
class Event(object):
    def __init__(self):
        self.primary = None
        self.energy = None
        self.zenith = None
        self.azimuth = None
        self.coreX = None
        self.coreY = None

        self.hasScintInfo = False
        self.nScints = None
        self.lgSref = None
        self.ldfScintSlope = None
        self.psiScint = None
        self.dCoreScint = None

        self.hasAntInfo = False
        self.nAnts = None
        self.psiAnt = None
        self.dCoreAnt = None

        self.pAnyMuon = None
        self.pHighEMuon = None
        self.eHighMuObslev = None
        self.n500GeVMuObslev = None
        self.nInIceMuons = None
        self.eDepInIce = None
        self.eMuInIce = None
        self.eMu1TeVInIce = None
        
        self.nMuonsObslev = None

        self.nThrows = None
        self.throwID = None
        self.effArea = None

        self.xmax = None
        self.nEmAtXmax = None
        
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
        
        self.nEmObslev = None

        self.nMu50m = None
        self.nMu100m = None
        self.nMu150m = None
        self.nMu200m = None
        self.nMu250m = None
        self.nMu300m = None
        self.nMu350m = None
        self.nMu400m = None
        self.nMu450m = None
        self.nMu500m = None
        self.nMu550m = None
        self.nMu600m = None
        self.nMu650m = None
        self.nMu700m = None
        self.nMu750m = None
        self.nMu800m = None
        self.nMu850m = None
        self.nMu900m = None
        self.nMu950m = None
        self.nMu1000m = None

        self.nEM50m = None
        self.nEM100m = None
        self.nEM150m = None
        self.nEM200m = None
        self.nEM250m = None
        self.nEM300m = None
        self.nEM350m = None
        self.nEM400m = None
        self.nEM450m = None
        self.nEM500m = None
        self.nEM550m = None
        self.nEM600m = None
        self.nEM650m = None
        self.nEM700m = None
        self.nEM750m = None
        self.nEM800m = None
        self.nEM850m = None
        self.nEM900m = None
        self.nEM950m = None
        self.nEM1000m = None


class EventProfileFit(object):
    def __init__(self):
        self.primary = None
        self.energy = None
        self.zenith = None
        self.azimuth = None
        
        self.nMuObslevLong = None
        self.nEmObslevLong = None
        self.nEmXmaxLong = None
        
        self.Rcorsika = None
        self.Lcorsika = None
        self.xmaxCorsika = None
        
        self.Rfit = None
        self.Lfit = None
        self.Xmaxfit = None
        self.sigmaRfit = None
        self.sigmaLfit = None
        self.sigmaXmaxfit = None

        self.RfitAndringa = None
        self.LfitAndringa = None
        self.XmaxfitAndringa = None
        self.sigmaRfitAndringa = None
        self.sigmaLfitAndringa = None
        self.sigmaXmaxfitAndringa = None
