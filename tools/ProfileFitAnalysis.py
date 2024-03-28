# Use this file for plotting contours, projection plots, and FOM plots for multivariate mass sensitivity analysis

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import rc, rcParams

rc("font", size=18.0)


from scipy import interpolate, stats

import random
import numpy as np

from .ASCIIFileReader import Event
from .PlottingTools import qualitative_colors


class MultivariateMassAnalysis(object):

    allObservables = [r"$X_{\rm max}$ (g/cm$^2$)",
                      r"lg(N$_{{\rm e, max}}$)", r"lg(N$_{{\rm e}}$)", "R", r"L (g/cm$^2$)"]

    allObservablesNoScaling = [r"$X_{\rm max, true}$ (g/cm$^2$)",
                               r"lg(N$_{{\rm e, max}}$)", r"lg(N$_{{\rm e, true}}$)", r"R$_{\rm true}$", r"L$_{\rm true}$ (g/cm$^2$)"]

    allPlottingNames = ["Xmax",
                        "nEM", "nEMObslev", "Rval", "Lval"]

    # If color is "#555555" then this color needs to be updated in the future...
    observableColors = ["#88CCEE",
                        "#555555", "#555555", "#117733", "#44AA99"]


    def __init__(self, minDeg=0, maxDeg=72, muonScaling=0.0, highEmuonScaling=0.0, minLgE=16, maxLgE=18.5,
                 includeXmax=False, includeEMxmax=False, includeEMObslev=False,
                 includeRval=False, includeLval=False, useGHFits=False, useCorsikaXmax=False,
                 allfourPrimaries=False,
                 protonAndHelium=False, heliumAndOxygen=False, applyScaling=True, applyDataCuts=False,
                 observatory="IceCube", useLargerSmearValues=False, singleObservable=False, smearVal=0.0): 

        # ====================================================================== #
        # Error handling for if conflicting keywords are set + Warnings for user #
        # ====================================================================== #

        if observatory != "IceCube" and observatory != "Auger":
            # If I do this then I should get rid of the muonScaling and highEmuonScaling keywords and instead put a type of scaling keyword 
            # i.e. scalingType={"energy", "EMxmax", "EMobslev"}
            # This may be too ambitious to do at the moment and maybe should be put on the back-burner for now...
            raise ValueError("Observatory must be set to either IceCube or Auger location! Other observatory location scalings can be added in the XXXXX file.")

        if allfourPrimaries + protonAndHelium + heliumAndOxygen > 1:
            raise ValueError("Only one keyword specifying primary particle types can be set to True.")

        if muonScaling == 0 or highEmuonScaling == 0:
            print("Warning: Only one muon scaling factor is set.")
            #raise ValueError("Must set both muon scaling factors (even if only using one type).")

        if applyScaling == False:
            print("Warning: The muon observables will not be scaled according to the Heitler-Matthews model!")

        if applyDataCuts == False:
            print("Warning: Anomolous profile fits will not be excluded from any resulting analysis or plots!")

        self.observatoryName = observatory
        self.muonEnergyScaling = muonScaling
        self.highEmuonEnergyScaling = highEmuonScaling

        self.minLgE = minLgE
        self.maxLgE = maxLgE

        self.minDeg = minDeg
        self.minZen = self.ZenithScaling(self.minDeg / 180.0 * np.pi)
        self.maxDeg = maxDeg
        self.maxZen = self.ZenithScaling(self.maxDeg / 180.0 * np.pi)

        self.flagEMxmax = includeEMxmax
        self.flagEMObslev = includeEMObslev
        self.flagRval = includeRval
        self.flagLval = includeLval
        self.flagGHFits = useGHFits
        self.flagCorsikaXmax = useCorsikaXmax

        self.flagAllPrimaries = allfourPrimaries
        self.flagProtonHelium = protonAndHelium
        self.flagHeliumOxygen = heliumAndOxygen

        self.flagScalingCorrections = applyScaling
        self.flagDataCuts = applyDataCuts
        self.flagLargeSmearUncerts = useLargerSmearValues
        self.flagSingleObservable = singleObservable


        if self.flagAllPrimaries:
            self.primaryNames = {"2212": "Proton", "1000020040": "Helium", "1000080160": "Oxygen", "1000260560": "Iron"}
            self.primaryColors = qualitative_colors(4)[::-1]
        elif self.flagProtonHelium:
            self.primaryNames = {"2212": "Proton", "1000020040": "Helium"}
            self.primaryColors = qualitative_colors(4)[::-1][0:2]
        elif self.flagHeliumOxygen:
            self.primaryNames = {"1000020040": "Helium", "1000080160": "Oxygen"}
            self.primaryColors = qualitative_colors(4)[::-1][1:3]
        else:
            self.primaryNames = {"2212": "Proton", "1000260560": "Iron"}
            self.primaryColors = qualitative_colors(4)[::-3]


        if self.flagSingleObservable == True:
            self.smearVal = smearVal
            print("Warning: Only one observable will be used in the analysis. If making contour/projection plots an error will occur!")


        self.kwObservables = [includeXmax,
                              includeEMxmax, includeEMObslev, includeRval, includeLval]

        self.params = []
        self.colors = []
        self.observableIndices = []
        self.plotNames = []

        for index in range(len(self.kwObservables)):
            if self.kwObservables[index] == 1:
                self.colors.append(self.observableColors[index])
                self.observableIndices.append(index)
                self.plotNames.append(self.allPlottingNames[index])
                if self.flagScalingCorrections == True:
                    self.params.append(self.allObservables[index])
                else:
                    self.params.append(self.allObservablesNoScaling[index])

        if len(self.params) == 0:
            raise ValueError("No observables were included in the analysis with this use of keywords. Try again with a different combination.")

        if singleObservable == True and len(self.params) != 1:
            raise ValueError("'singleObservable' keyword is set but the more than one observable was passed to the observables list.")

        self.data = {}
        for key in self.primaryNames.keys():
            self.data[self.primaryNames[key]] = [[] for i in range(len(self.params))]

        self.sigmas = [0.9, 0.68]  # Contour line definitions (solely used for contour plots)

        self.eventList = []

    def ZenithScaling(self, z):
        return np.sin(z) ** 2

    def ReadSingleFile(self, file):

        with open(file, "r") as file:
            for line in file:
                if line[0] == "#":
                    continue

                cols = line.split()

                if len(cols) != 71:
                    continue

                azimuth = float(cols[3]) # In units of radians
                zenith = float(cols[2]) # In units of radians

                if not self.minDeg * (np.pi / 180.0) < zenith < self.maxDeg * (np.pi / 180.0):
                    continue

                energy = float(cols[1]) # In units of GeV
                if energy == 0.:
                    continue

                if not self.minLgE < np.log10(energy * 1e+9) < self.maxLgE:
                    continue

                event = Event()

                particleID = int(cols[0])
                
                if self.flagAllPrimaries:
                    corsikaIDs = [14, 402, 1608, 5626]
                    pdgIDs = [2212, 1000020040, 1000080160, 1000260560]
                elif self.flagProtonHelium:
                    corsikaIDs = [14, 402]
                    pdgIDs = [2212, 1000020040]
                elif self.flagHeliumOxygen:
                    corsikaIDs = [402, 1608]
                    pdgIDs = [1000020040, 1000080160]
                else:
                    corsikaIDs = [14, 5626]
                    pdgIDs = [2212, 1000260560]
                
                if particleID in corsikaIDs:
                    particleID = pdgIDs[corsikaIDs.index(particleID)]
                else:
                    continue

                event.primary = particleID

                event.energy = energy
                event.zenith = zenith
                event.azimuth = azimuth

                # Test...
                # If using, should also change highE muon name in 'allObservables' list!
                #event.n500GeVMuObslev = float(cols[5]) # Number of muons at ground w/ E > 300 GeV (IC), w/ E > 1 GeV (Auger)
                #event.n500GeVMuObslev = float(cols[7]) # Number of muons at ground w/ E > 1 TeV = 1000 GeV, use scaling of 0.82 instead of 0.83...

                event.xmax = float(cols[53])
                event.n500GeVMuObslev = float(cols[6])
                event.nMuonsObslev = float(cols[4]) # Number of muons at ground (at Obslev)
                event.nEmAtXmax = float(cols[55])
                event.nEmObslev = float(cols[32]) # Number of electrons/positrons at ground (at Obslev)

                event.Rcorsika = float(cols[57]) # Shape parameter of corsika GH fit from .long file
                event.Lcorsika = float(cols[58]) # Characteristic width of corsika GH fit from .long file
                event.Rfit = float(cols[59]) # R from GH shifted fit to .long file 
                event.Lfit = float(cols[61]) # L from GH shifted fit to .long file
                event.sigmaRfit = float(cols[60]) # Uncertainty in R from GH shifted fit
                event.sigmaLfit = float(cols[62]) # Uncertainty in L from GH shifted fit

                event.Xmaxfit = float(cols[63]) # Xmax from GH shifted fit to .long file
                event.sigmaXmaxfit = float(cols[64]) # Uncertainty in Xmax from GH shifted fit

                event.RfitAndringa = float(cols[65]) # R from Andringa fit to .long file 
                event.LfitAndringa = float(cols[67]) # L from Andringa fit to .long file
                event.sigmaRfitAndringa = float(cols[66]) # Uncertainty in R from Andringa fit
                event.sigmaLfitAndringa = float(cols[68]) # Uncertainty in L from Andringa fit

                event.XmaxfitAndringa = float(cols[69]) # Xmax from Andringa fit to .long file
                event.sigmaXmaxfitAndringa = float(cols[70]) # Uncertainty in Xmax from Andringa fit

                event.nMu50m = float(cols[12])
                event.nMu100m = float(cols[13])
                event.nMu150m = float(cols[14])
                event.nMu200m = float(cols[15])
                event.nMu250m = float(cols[16])
                event.nMu300m = float(cols[17])
                event.nMu350m = float(cols[18])
                event.nMu400m = float(cols[19])
                event.nMu450m = float(cols[20])
                event.nMu500m = float(cols[21])
                event.nMu550m = float(cols[22])
                event.nMu600m = float(cols[23])
                event.nMu650m = float(cols[24])
                event.nMu700m = float(cols[25])
                event.nMu750m = float(cols[26])
                event.nMu800m = float(cols[27])
                event.nMu850m = float(cols[28])
                event.nMu900m = float(cols[29])
                event.nMu950m = float(cols[30])
                event.nMu1000m = float(cols[31])

                event.nEM50m = float(cols[33])
                event.nEM100m = float(cols[34])
                event.nEM150m = float(cols[35])
                event.nEM200m = float(cols[36])
                event.nEM250m = float(cols[37])
                event.nEM300m = float(cols[38])
                event.nEM350m = float(cols[39])
                event.nEM400m = float(cols[40])
                event.nEM450m = float(cols[41])
                event.nEM500m = float(cols[42])
                event.nEM550m = float(cols[43])
                event.nEM600m = float(cols[44])
                event.nEM650m = float(cols[45])
                event.nEM700m = float(cols[46])
                event.nEM750m = float(cols[47])
                event.nEM800m = float(cols[48])
                event.nEM850m = float(cols[49])
                event.nEM900m = float(cols[50])
                event.nEM950m = float(cols[51])
                event.nEM1000m = float(cols[52])

                self.eventList.append(event)

    def GetValues(self):

        prevZen = 0
        prevAzi = 0

        if not len(self.eventList):
            print("No events were loaded which passed the cuts!")

        for event in self.eventList:
            # Apply cut for xmax because very large xmax values are unphysical

            if not 0 < event.xmax < 1500:
                continue

            # Apply data cuts if keyword provided...
            if (self.flagDataCuts == True) and (self.flagGHFits == True):
                if event.sigmaXmaxfit == np.inf or event.sigmaRfit == np.inf or event.sigmaLfit == np.inf:
                    continue
                elif event.sigmaXmaxfit > 5.0 or event.sigmaRfit > 0.05 or event.sigmaLfit > 5.0:
                    continue
                elif event.Xmaxfit < 0.0:  # Maybe also include a cut on L values? (i.e. L < 350 or L < 325???)
                    continue
            elif (self.flagDataCuts == True) and (self.flagGHFits == False):
                if event.sigmaXmaxfitAndringa == np.inf or event.sigmaRfitAndringa == np.inf or event.sigmaLfitAndringa == np.inf:
                    continue
                elif event.sigmaXmaxfitAndringa > 5.0 or event.sigmaRfitAndringa > 0.05 or event.sigmaLfitAndringa > 5.0:
                    continue
                elif event.RfitAndringa < 0.0 or event.LfitAndringa > 350.0:  # Maybe also include a cut on L values? (i.e. L < 350 or L < 325???)
                    continue

            zen = event.zenith
            azi = event.azimuth

            # Do not repeat showers
            if zen == prevZen and azi == prevAzi:
                continue

            name = self.primaryNames[str(event.primary)]

            energy = event.energy


            if self.flagScalingCorrections == False:
                lgNEM = np.log10(event.nEmAtXmax)

                # Changed name to have "Corr" at the end to remain consistent w/ other observables, made it easier to implement in code...
                lgNEMObslevCorr = np.log10(event.nEmObslev)

            else:
                if self.muonEnergyScaling == 0.94 and self.highEmuonEnergyScaling == 0.82:
                    if self.observatoryName == "Auger":
                        raise ValueError("This scaling has not been optimized for the Auger observatory. Repeat scaling steps for Auger and then update code to continue.")

                    lgNEM = np.log10(event.nEmAtXmax / (energy * 1e-9) ** 1.01)

                    lgNEMObslev = np.log10(event.nEmObslev / (energy * 1e-9) ** 1.15)

                elif self.muonEnergyScaling == 0.93 and self.highEmuonEnergyScaling == 0.82:
                    # For these parameters then scale w.r.t. the electron number at Xmax
                    if self.observatoryName == "IceCube":
                        scaleCorrection = 0.01 # Correction between lg(Ne) vs. lg(E) plot
                        EeVnEMNormalization = 605741418.2773747 # zen = 0-72 deg (all zenith angles), lgE = 17.9-18.1
                    elif self.observatoryName == "Auger":
                        scaleCorrection = 0.01 # Correction between lg(Ne) vs. lg(E) plot
                        EeVnEMNormalization = 586908936.4969574 # zen = 0-65 deg (Auger, all zenith angles), lgE = 17.9-18.1

                    lgNEM = np.log10(event.nEmAtXmax)

                    if self.observatoryName == "IceCube":
                        lgNEMObslevCorr = np.log10(event.nEmObslev / (event.nEmAtXmax / EeVnEMNormalization) ** (1.13 + scaleCorrection))
                    elif self.observatoryName == "Auger":
                        lgNEMObslevCorr = np.log10(event.nEmObslev / (event.nEmAtXmax / EeVnEMNormalization) ** (1.16 + scaleCorrection))


                elif self.muonEnergyScaling == 0.81 and self.highEmuonEnergyScaling == 0.72:
                    raise ValueError("This scaling type does not normalize the electron numbers at ground and therefore more scaling analysis is necessary.")

                    if self.observatoryName == "Auger":
                        raise ValueError("This scaling has not been optimized for the Auger observatory. Repeat scaling steps for Auger and then update code to continue.")

                    # For these parameters then scale w.r.t. the electron number at observation level (i.e. at ground)
                    scaleCorrection = 0.15 # Correction between lg(Ne) at Ground vs. lg(E) plot
                    #scaleCorrection = 0.01 # Correction to make it so everything scales correctly vs. energy (not sure why this value works...)

                    lgNEM = np.log10(event.nEmAtXmax / (event.nEmObslev) ** (0.87 + scaleCorrection))

                    lgNEMObslev = np.log10(event.nEmObslev / (event.nEmObslev) ** (0.99 + scaleCorrection))

                else:
                    raise ValueError("Not a valid combination of muon energy scaling factors...")


            if self.flagGHFits == True and self.flagCorsikaXmax == True:
                Xmaxval = event.xmax
                Rval = event.Rfit
                Lval = event.Lfit
                raise ValueError("This combination of flags for Xmax, R, and L has not scaled! Update the code to acount for this scaling.")
            elif self.flagGHFits == True and self.flagCorsikaXmax == False:
                Xmaxval = event.Xmaxfit
                Rval = event.Rfit
                Lval = event.Lfit
                raise ValueError("This combination of flags for Xmax, R, and L has not scaled! Update the code to acount for this scaling.")
            elif self.flagGHFits == False and self.flagCorsikaXmax == True:                
                Xmaxval = event.xmax
                Rval = event.RfitAndringa
                Lval = event.LfitAndringa
                raise ValueError("This combination of flags for Xmax, R, and L has not scaled! Update the code to acount for this scaling.")
            else:
                if self.flagScalingCorrections == False:
                    XmaxvalCorr = event.XmaxfitAndringa
                    RvalCorr = event.RfitAndringa
                    LvalCorr = event.LfitAndringa
                else:
                    if self.observatoryName == "IceCube":
                        XmaxvalCorr = event.XmaxfitAndringa - (62.01 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        RvalCorr = event.RfitAndringa - (-0.03 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        LvalCorr = event.LfitAndringa - (7.18 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                    elif self.observatoryName == "Auger":
                        XmaxvalCorr = event.XmaxfitAndringa - (62.82 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        RvalCorr = event.RfitAndringa - (-0.03 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)
                        LvalCorr = event.LfitAndringa - (7.47 + scaleCorrection)*np.log10(event.nEmAtXmax / EeVnEMNormalization)

                        if XmaxvalCorr == np.nan or XmaxvalCorr == np.inf:
                            print(f"Bad value found! With Xmax={event.XmaxfitAndringa}, EMatXmax={event.nEmAtXmax}")


            allValues = [XmaxvalCorr,
                         lgNEM, lgNEMObslevCorr, RvalCorr, LvalCorr]

            vals = [allValues[ind] for ind in self.observableIndices]

            if len(vals) == 0:
                raise ValueError("No observables were included in the analysis with this use of keywords. Try again with a different combination.")

            if self.flagSingleObservable == True and len(vals) != 1:
                raise ValueError("'singleObservable' keyword is set but more than one observable was saved to the 'vals' observable list.")

            for ival, val in enumerate(vals):
                self.data[name][ival].append(val)


            prevZen = zen
            prevAzi = azi

        # Convert to number arrays for later
        for key in self.data.keys():
            for i in range(len(self.data[key])):
                self.data[key][i] = np.array(self.data[key][i])

    def SmearValues(self, vals):
        """
        Smears the values for adding measurement uncertainty
        """

        if self.flagScalingCorrections == False:
            raise ValueError("Observables not scaled, will not smear unscaled values...")
        elif self.flagLargeSmearUncerts == True:
            if self.muonEnergyScaling == 0.94 and self.highEmuonEnergyScaling == 0.82:
                XmaxSmear = 40.0
                EMXmaxSmear = 0.1
                EMObslevSmear = 0.2
                RSmear = 0.1  # double uncertainty from arxiv: 1811.04660
                LSmear = 10.0  # in arxiv: 1811.04660 list uncertainty as 7.3+0.9 add in quad
            elif self.muonEnergyScaling == 0.93 and self.highEmuonEnergyScaling == 0.82:
                XmaxSmear = 40.0
                EMXmaxSmear = 0.1  # This is smearing lgNe,max not Ne,max --> sigma of Ne,max is 0.1 so sigma of lgNe,max is ~0.045
                EMObslevSmear = 0.2  # ~21-26% uncertainty in raw value
                RSmear = 0.1  # double uncertainty from arxiv: 1811.04660
                LSmear = 10.0  # in arxiv: 1811.04660 list uncertainty as 7.3+0.9 add in quad
            elif self.muonEnergyScaling == 0.81 and self.highEmuonEnergyScaling == 0.72:                
                XmaxSmear = 40.0
                EMXmaxSmear = 0.1
                EMObslevSmear = 0.2
                RSmear = 0.1  # double uncertainty from arxiv: 1811.04660
                LSmear = 10.0  # in arxiv: 1811.04660 list uncertainty as 7.3+0.9 add in quad
            else:
                raise ValueError("Not a valid combination of muon energy scaling factors...")            
        else:
            if self.muonEnergyScaling == 0.94 and self.highEmuonEnergyScaling == 0.82:
                XmaxSmear = 20.0
                EMXmaxSmear = 0.05
                EMObslevSmear = 0.1
                RSmear = 0.05  # in arxiv: 1811.04660 list uncertainty as 0.04+0.012 add in quad
                LSmear = 5.0  # half of uncertainty from arxiv: 1811.04660
            elif self.muonEnergyScaling == 0.93 and self.highEmuonEnergyScaling == 0.82:
                XmaxSmear = 20.0
                EMXmaxSmear = 0.05  # This is smearing lgNe,max not Ne,max --> sigma of Ne,max is 0.1 so sigma of lgNe,max is ~0.045
                EMObslevSmear = 0.1  # ~21-26% uncertainty in raw value
                RSmear = 0.05  # in arxiv: 1811.04660 list uncertainty as 0.04+0.012 add in quad
                LSmear = 5.0  # half of uncertainty from arxiv: 1811.04660
            elif self.muonEnergyScaling == 0.81 and self.highEmuonEnergyScaling == 0.72:                
                XmaxSmear = 20.0
                EMXmaxSmear = 0.05
                EMObslevSmear = 0.1
                RSmear = 0.05  # in arxiv: 1811.04660 list uncertainty as 0.04+0.012 add in quad
                LSmear = 5.0  # half of uncertainty from arxiv: 1811.04660
            else:
                raise ValueError("Not a valid combination of muon energy scaling factors...")


        if self.flagSingleObservable:
            vals[0] += stats.norm.rvs(loc=0.0, scale=self.smearVal)
        else:

            allSmearVals = [XmaxSmear,
                            lowEMuTotSmear, RatioEMNMuTot, RatioEMObslevNMuTot,
                            highEMuSmear, RatioEMNMuHigh, RatioEMObslevNMuHigh,
                            lowEMu800Smear, RatioEMNMu850m, RatioEMObslevNMu850m,
                            EMXmaxSmear, EMObslevSmear, RSmear, LSmear]

            SmearVals = [allSmearVals[ind] for ind in self.observableIndices]

            for i in range(len(vals)):
                vals[i] += stats.norm.rvs(loc=0.0, scale=SmearVals[i])


    def GetContoursForPlot(self, xVals, yVals, xBins, yBins):

        counts, xBins, yBins = np.histogram2d(xVals, yVals, bins=(xBins, yBins))

        countsNorm = counts / counts.sum()

        n = 1000
        t = np.linspace(0, countsNorm.max(), n)
        integral = ((countsNorm >= t[:, None, None]) * countsNorm).sum(axis=(1, 2))
        f_interp = interpolate.interp1d(integral, t)
        t_contours = f_interp(np.array(self.sigmas))

        return xBins, yBins, counts, countsNorm, t_contours


    def MakeContourPlots(self, filename, showRawHistograms=False):

        fileSplit = filename.rsplit("/", 1)
        fileNameString = fileSplit[-1].rsplit(".", 1)

        dumpFig = plt.figure()
        dumpAx = dumpFig.add_subplot(1, 1, 1)

        nRows = 1
        nCols = 1

        gs = gridspec.GridSpec(nRows, nCols, wspace=0.3, hspace=0.3)

        for ipar in range(len(self.params)):
            for jpar in range(ipar + 1, len(self.params)):

                fig = plt.figure(figsize=(nCols * 18.0 / 2.54, nRows * 15.0 / 2.54))
                ax = fig.add_subplot(gs[0])
                ax.set_xlabel(self.params[ipar])
                ax.set_ylabel(self.params[jpar])

                print("Comparing", self.params[ipar], "and", self.params[jpar])

                minX = 1e100
                maxX = -1e100
                minY = 1e100
                maxY = -1e100

                nBins = 10

                for primkey in self.data.keys():
                    prim = self.data[primkey]
                    if not len(prim[ipar]):
                        print("Primary", primkey, "has no entries for", self.params[ipar])
                        continue
                    minX = min([minX, min(prim[ipar])])
                    maxX = max([maxX, max(prim[ipar])])
                    minY = min([minY, min(prim[jpar])])
                    maxY = max([maxY, max(prim[jpar])])
                    nBins = int(np.sqrt(len(prim[ipar])) * 0.66)

                dX = abs(minX - maxX)
                minX -= dX * 0.1
                maxX += dX * 0.1
                dY = abs(minY - maxY)
                minY -= dY * 0.1
                maxY += dY * 0.1

                xBins = np.linspace(minX, maxX, nBins)
                yBins = np.linspace(minY, maxY, nBins)

                points = [[[] for i in range(len(self.sigmas))] for j in range(len(self.data))]

                for iprim, prim in enumerate(self.data.keys()):
                    xVals = self.data[prim][ipar]
                    yVals = self.data[prim][jpar]

                    if not len(xVals) or not len(yVals):
                        continue

                    xBins, yBins, counts, contourLoc, contourZ = self.GetContoursForPlot(xVals, yVals, xBins, yBins)

                    if showRawHistograms == True:
                        # Need to check if this still works...
                        x, y = np.meshgrid(xBins, yBins)
                        plt.pcolormesh(x, y, np.log10(counts.T))

                    cs = dumpAx.contour(contourLoc.T, contourZ, extent=[minX, maxX, minY, maxY])

                    for isig in range(len(self.sigmas)):
                        for p in cs.collections[len(self.sigmas) - 1 - isig].get_paths():
                            v = p.vertices
                            x = v[:, 0]
                            y = v[:, 1]

                            points[iprim][isig].append([x, y])

                style = ["-", "--", ":"]

                for isig in range(len(self.sigmas)):
                    for iprim, prim in enumerate(self.data.keys()):

                        # Get the lengths so that you can ignore the small outlying islands
                        lens = [len(lineSet[0]) for lineSet in points[iprim][isig]]

                        for lineSet in points[iprim][isig]:
                            if len(lineSet[0]) != max(lens):  # ignore the little islands
                                continue

                            ax.plot(lineSet[0], lineSet[1], lw=2.5, linestyle=style[isig % len(style)], color=self.primaryColors[iprim])
                            ax.fill(lineSet[0], lineSet[1], color=self.primaryColors[iprim], alpha=0.2 / len(self.sigmas))

                    ax.plot([], [], linestyle=style[isig % len(style)], lw=2.5, label="{}%".format(int(self.sigmas[len(self.sigmas) - 1 - isig] * 100)), color="k")

                for iprim, prim in enumerate(self.data.keys()):
                    ax.plot([], [], lw=2.5, label=prim, color=self.primaryColors[iprim])

                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()

                ax.set_xlim(xmin, xmin + (xmax - xmin) * 1.0)
                ax.set_ylim(ymin, ymin + (ymax - ymin) * 1.0)
                ax.legend(loc="upper right", prop={"size": 14})

                ax.tick_params(direction="in", which="both", axis="both")
                ax.yaxis.set_ticks_position("both")
                ax.xaxis.set_ticks_position("both")

                yName = self.plotNames[jpar]
                xName = self.plotNames[ipar]

                if self.observatoryName == "IceCube":
                    ax.text(0.40, 0.93, "IceCube", transform=ax.transAxes, fontsize=18)
                elif self.observatoryName == "Auger":
                    ax.text(0.40, 0.93, "Auger", transform=ax.transAxes, fontsize=18)

                # Can add text to contour plots showing zenith and energy ranges
                # Exclude for now as would need to do this on case-by-case basis depending on observables studied
                #if yName == "Lval" and xName == "Xmax" and self.observatoryName == "IceCube":
                #    ax.text(0.48, 0.11, rf"$\theta_{{\rm zen}} = {self.minDeg:.0f}^{{\circ}}-{self.maxDeg:.0f}^{{\circ}}$", transform=ax.transAxes, fontsize=18)
                #    ax.text(0.48, 0.04, r"E = $10^{16.5}-10^{16.9}$ eV", transform=ax.transAxes, fontsize=18)
                #else:
                #    print("No need to include text in plot...")                    

                fileToSave = fileSplit[0] + "/contours/" + fileNameString[0] + f"_Contours_{yName}_{xName}.pdf"
                fig.savefig(fileToSave, bbox_inches="tight")
                print("Saved", fileToSave)



    ###################################################################################################
    # The class functions after this point were added specifically to analyze the shower profile fits # 
    ###################################################################################################

    def FindMeanMedianMode(self, valsArray, roundValsForMode=False):

        # Compute mean and median removing NaN's (to get true mean/median estimate)
        meanVal = np.nanmean(valsArray)
        medVal = np.nanmedian(valsArray)

        if roundValsForMode == True:
            roundVals = np.round(valsArray, 1)
            uniqueVals, counts = np.unique(roundVals, return_counts=True)
        else:
            uniqueVals, counts = np.unique(valsArray, return_counts=True)

        modeLocs = np.argwhere(counts == np.max(counts))

        modeVal = uniqueVals[modeLocs].flatten().tolist()

        if len(modeVal) == len(valsArray):
            modeVal = "None"
        elif len(modeVal) < len(valsArray) and len(modeVal) != 1:
            modeVal = "Multiple"
        else:
            modeVal = modeVal[0]

        return meanVal, medVal, modeVal


    def MakeHistograms(self, filename, fixedBins=False, calcMeanMedianMode=False):
        """
        Used to make histograms of the observable values from the shower profile fits (but no other observables)
        """

        nCols = 1
        nRows = 1
        gs = gridspec.GridSpec(nRows, nCols, wspace=0.3, hspace=0.3)

        for ipar in range(len(self.params)):
            if self.params[ipar] not in [r"$X_{\rm max}$ (g/cm$^2$)", "R", r"L (g/cm$^2$)"]:
                continue

            fig = plt.figure(figsize=(nCols * 18.0 / 2.54, nRows * 15.0 / 2.54))
            ax = fig.add_subplot(gs[0])
            ax.set_xlabel(f'{self.params[ipar]}')
            ax.set_ylabel('Counts')

            if fixedBins == False:
                # Variable bin range based on min/max values of the parameter
                minBin = 1000
                maxBin = 0

                for iprim, prim in enumerate(self.data.keys()):
                    valsForHist = self.data[prim][ipar]
            
                    if min(valsForHist) <= minBin:
                        minBin = min(valsForHist)

                    if max(valsForHist) >= maxBin:
                        maxBin = max(valsForHist)
            else:
                if self.params[ipar] == r"$X_{\rm max}$ (g/cm$^2$)":
                    minBin = 500
                    maxBin = 1100
                elif self.params[ipar] == "R":
                    minBin = 0.0
                    maxBin = 0.6
                elif self.params[ipar] == r"L (g/cm$^2$)":
                    minBin = 175
                    maxBin = 350
                else:
                    raise ValueError(f"Code not optimized for parameter {self.params[ipar]}. Try running with different parameters or updating code.")
            
            for iprim, prim in enumerate(self.data.keys()):
                valsForHist = self.data[prim][ipar]

                if calcMeanMedianMode == False:
                    yPrim, xPrim, junk = ax.hist(valsForHist, bins=np.linspace(minBin, maxBin, 50),
                                                 histtype='step', linewidth=2, color=self.primaryColors[iprim], label=prim)
                else:
                    meanVal, medVal, modeVal = self.FindMeanMedianMode(valsForHist, roundValsForMode=True)

                    if type(modeVal) != str:
                        modeVal = np.round(modeVal, 1)

                    labelString = f"{prim}, Mean: {meanVal:.1f}" + "\n" + f"Med: {medVal:.1f}, Mode: {modeVal}"

                    yPrim, xPrim, junk = ax.hist(valsForHist, bins=np.linspace(minBin, maxBin, 50),
                                                 histtype='step', linewidth=2, color=self.primaryColors[iprim], label=labelString)

            if calcMeanMedianMode == False:
                ax.legend(loc="upper right", prop={"size": 14})
            else:
                ax.legend(loc="upper right", prop={"size": 10})

            ax.tick_params(direction="in", which="both", axis="both")
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_ticks_position("both")

            if self.params[ipar] == r"$X_{\rm max}$ (g/cm$^2$)":
                xName = "Xmax"
                ax.text(0.74, 0.66, self.observatoryName, transform=ax.transAxes, fontsize=14)
                ax.text(0.65, 0.61, rf"$\theta_{{\rm zen}} = {self.minDeg}^{{\circ}}-{self.maxDeg}^{{\circ}}$", transform=ax.transAxes, fontsize=14)
                ax.text(0.65, 0.55, rf"lg(E) = {self.minLgE}$-${self.maxLgE}", transform=ax.transAxes, fontsize=14)
                if self.flagCorsikaXmax == True:
                    fitInfo = "CorsikaXmax"
                elif self.flagCorsikaXmax == False and self.flagGHFits == True:
                    fitInfo = "GHShiftedXmax"
                else:
                    fitInfo = "AndringaXmax"
            elif self.params[ipar] == "R":
                xName = "Rval"
                ax.text(0.12, 0.93, self.observatoryName, transform=ax.transAxes, fontsize=14)
                ax.text(0.03, 0.87, rf"$\theta_{{\rm zen}} = {self.minDeg}^{{\circ}}-{self.maxDeg}^{{\circ}}$", transform=ax.transAxes, fontsize=14)
                ax.text(0.03, 0.81, rf"lg(E) = {self.minLgE}$-${self.maxLgE}", transform=ax.transAxes, fontsize=14)
                if self.flagGHFits == True:
                    fitInfo = "GHShiftedR"
                else:
                    fitInfo = "AndringaR"
            elif self.params[ipar] == r"L (g/cm$^2$)":
                xName = "Lval"
                ax.text(0.74, 0.66, self.observatoryName, transform=ax.transAxes, fontsize=14)
                ax.text(0.65, 0.61, rf"$\theta_{{\rm zen}} = {self.minDeg}^{{\circ}}-{self.maxDeg}^{{\circ}}$", transform=ax.transAxes, fontsize=14)
                ax.text(0.65, 0.55, rf"lg(E) = {self.minLgE}$-${self.maxLgE}", transform=ax.transAxes, fontsize=14)
                if self.flagGHFits == True:
                    fitInfo = "GHShiftedL"
                else:
                    fitInfo = "AndringaL"
            else:
                raise ValueError(f"Code not written for parameter {self.params[ipar]}. Please update or try running again with different settings.")


            if fixedBins == True:
                fileToSave = filename + f"_Histogram_FixedBinRange_{xName}.pdf"
            else:
                fileToSave = filename + f"_Histogram_{xName}" + fitInfo + ".pdf"

            fig.savefig(fileToSave, bbox_inches="tight")
            print("Saved", fileToSave)


    def CountNumberBadFits(self, primary):

        prevZen = 0
        prevAzi = 0

        poorGHFits = 0
        poorAndringaFits = 0

        totalEvents = 0

        for event in self.eventList:

            if not 0 < event.xmax < 1500:
                continue

            zen = event.zenith
            azi = event.azimuth

            # Do not repeat showers (same as done in GetValues() function)
            if zen == prevZen and azi == prevAzi:
                continue

            name = self.primaryNames[str(event.primary)]

            # Only count number of bad fits for the primary in question from function call
            if name != primary:
                continue

            totalEvents += 1

            if event.sigmaXmaxfit == np.inf or event.sigmaRfit == np.inf or event.sigmaLfit == np.inf:
                poorGHFits += 1
            elif event.sigmaXmaxfit > 5.0 or event.sigmaRfit > 0.05 or event.sigmaLfit > 5.0:
                poorGHFits += 1
            elif event.Xmaxfit < 0.0:
                poorGHFits += 1

            if event.sigmaXmaxfitAndringa == np.inf or event.sigmaRfitAndringa == np.inf or event.sigmaLfitAndringa == np.inf:
                poorAndringaFits += 1
            elif event.sigmaXmaxfitAndringa > 5.0 or event.sigmaRfitAndringa > 0.05 or event.sigmaLfitAndringa > 5.0:
                poorAndringaFits += 1
            elif event.RfitAndringa < 0.0:
                poorAndringaFits += 1

            prevZen = zen
            prevAzi = azi

        poorFitRatiosDict = dict([('GH Shift.', float(poorGHFits / totalEvents)),
                                  ('GH Param.', float(poorAndringaFits / totalEvents))
                                 ])

        return poorFitRatiosDict, totalEvents


    def PlotBadFitsFractions(self, path, includeBadFitValues=False):

        markers = ['o', 'd', 'P', '*']

        nCols = 1
        nRows = 1

        gs = gridspec.GridSpec(nRows, nCols, wspace=0.3, hspace=0.3)

        fig = plt.figure(figsize=(nCols * 18.0 / 2.54, nRows * 15.0 / 2.54))
        ax = fig.add_subplot(gs[0])
        ax.set_ylabel('Percent of Poor Fits [%]')

        for iprim, prim in enumerate(self.data.keys()):

            ratiosDict, totEvents = self.CountNumberBadFits(prim)

            xLabels = list(ratiosDict.keys())
            xVals = np.linspace(0, len(xLabels)-1, len(xLabels))
            yVals = np.array(list(ratiosDict.values())) * 100.0 # Convert to a percentage

            if includeBadFitValues == True:
                totBadFits = (yVals / 100.0) * totEvents
                ax.scatter(xVals, yVals, marker=markers[iprim], c=self.primaryColors[iprim], label=f'{prim}, Tot. Fits = {totEvents}, Bad Fits = {totBadFits}')
            else:
                ax.scatter(xVals, yVals, marker=markers[iprim], c=self.primaryColors[iprim], label=f'{prim}, Tot. Fits = {totEvents}')

            if iprim == 0:            
                ax.set_xticks(xVals)
                ax.set_xticklabels(xLabels, minor=False, rotation=45)

        ax.legend(loc="best", prop={"size": 14})

        ax.yaxis.set_minor_locator(MultipleLocator(0.50))
        ax.tick_params(direction="in", which="both", axis="both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")

        if self.observatoryName == "IceCube":
            ax.text(0.40, 0.65, "IceCube", transform=ax.transAxes, fontsize=14)
        elif self.observatoryName == "Auger":
            ax.text(0.45, 0.65, "Auger", transform=ax.transAxes, fontsize=14)
        ax.text(0.60, 0.65, rf"$\theta_{{\rm zen}} = {self.minDeg}^{{\circ}}-{self.maxDeg}^{{\circ}}$", transform=ax.transAxes, fontsize=14)
        ax.text(0.60, 0.59, rf"lg(E) = {self.minLgE}$-${self.maxLgE}", transform=ax.transAxes, fontsize=14)

        ax.grid(axis='both', color='k', linestyle='-', alpha=0.3)

        if self.flagDataCuts == True:
            fileToSave = path + f"/plots/misc/" + self.observatoryName + f"_FractionOfPoorFits_zen{self.minDeg}_{self.maxDeg}_lgE{self.minLgE}_{self.maxLgE}_DataCutApplied.pdf"
        else:
            fileToSave = path + f"/plots/misc/" + self.observatoryName + f"_FractionOfPoorFits_zen{self.minDeg}_{self.maxDeg}_lgE{self.minLgE}_{self.maxLgE}.pdf"
        fig.savefig(fileToSave, bbox_inches="tight")
        print("Saved", fileToSave)


    def CountTypesBadFits(self, primary, fit=None):

        if fit not in ["GHShifted", "Andringa"]:
            raise ValueError("Must supply a valid fit type here. Valid types are 'GHShifted' or 'Andringa'.")

        prevZen = 0
        prevAzi = 0

        totalEvents = 0

        fitsUnconstrained = 0
        fitsXmaxAbove1500 = 0
        fitsXmaxBelow0 = 0
        fitsRAbove1 = 0
        fitsLAbove350 = 0
        fitsSigmaXmaxAbove5 = 0
        fitsSigmaRAbovePoint05 = 0
        fitsSigmaLAboveCut = 0

        for event in self.eventList:

            if not 0 < event.xmax < 1500:
                continue

            zen = event.zenith
            azi = event.azimuth

            # Do not repeat showers (same as done in GetValues() function)
            if zen == prevZen and azi == prevAzi:
                continue

            name = self.primaryNames[str(event.primary)]

            # Only count number of bad fits for the primary in question from function call
            if name != primary:
                continue

            totalEvents += 1

            if fit == "GHShifted":
                sigXmax = event.sigmaXmaxfit
                sigR = event.sigmaRfit
                sigL = event.sigmaLfit
                XmaxValue = event.Xmaxfit
                RValue = event.Rfit
                LValue = event.Lfit
                sigLString = 'sigmaL>5'
                sigLCut = 5.0
            elif fit == "Andringa":
                sigXmax = event.sigmaXmaxfitAndringa
                sigR = event.sigmaRfitAndringa
                sigL = event.sigmaLfitAndringa
                XmaxValue = event.XmaxfitAndringa
                RValue = event.RfitAndringa
                LValue = event.LfitAndringa
                sigLString = 'sigmaL>5'
                sigLCut = 5.0

            if sigXmax == np.inf or sigR == np.inf or sigL == np.inf:
                fitsUnconstrained += 1
                continue

            if XmaxValue > 1500.0:
                fitsXmaxAbove1500 += 1
                continue

            if XmaxValue < 0.0:
                fitsXmaxBelow0 += 1
                continue

            if RValue > 1.0:
                fitsRAbove1 += 1
                continue

            if LValue > 350.0:
                fitsLAbove350 += 1
                continue

            if sigXmax > 5.0:
                fitsSigmaXmaxAbove5 += 1
                continue

            if sigR > 0.05:
                fitsSigmaRAbovePoint05 += 1
                continue

            if sigL > sigLCut:
                fitsSigmaLAboveCut += 1
                continue

            prevZen = zen
            prevAzi = azi


        typesBadFitRatiosDict = dict([('Total Fits', float(totalEvents)),
                                  ('Unconstrained', float(fitsUnconstrained)),
                                  ('Xmax>1500', float(fitsXmaxAbove1500)),
                                  ('Xmax<0', float(fitsXmaxBelow0)),
                                  ('R>1', float(fitsRAbove1)),
                                  ('L>350', float(fitsLAbove350)),
                                  ('sigmaXmax>5', float(fitsSigmaXmaxAbove5)),
                                  ('sigmaR>.05', float(fitsSigmaRAbovePoint05)),
                                  (sigLString, float(fitsSigmaLAboveCut))
                                 ])

        return typesBadFitRatiosDict


    def PlotTypesBadFits(self, path, fittype=None):

        if fittype not in ["GHShifted", "Andringa"]:
            raise ValueError("Must supply a valid fit type here. Valid types are 'GHShifted' or 'Andringa'.")

        nCols = 1
        nRows = 1
        gs = gridspec.GridSpec(nRows, nCols, wspace=0.3, hspace=0.3)

        fig = plt.figure(figsize=(nCols * 18.0 / 2.54, nRows * 15.0 / 2.54))
        ax = fig.add_subplot(gs[0])
        ax.set_ylabel('Number Fits')

        for iprim, prim in enumerate(self.data.keys()):

            ratiosDict = self.CountTypesBadFits(prim, fit=fittype)

            xLabels = list(ratiosDict.keys())[1:]
            xVals = np.linspace(0, len(xLabels)-1, len(xLabels))
            yVals = np.array(list(ratiosDict.values())[1:]) #* 100.0 # Convert to a percentage

            totEvents = list(ratiosDict.values())[0]
            totBadFits = yVals.sum()

            ax.plot(xVals, yVals, linestyle='solid', linewidth=2, marker='o', color=self.primaryColors[iprim], label=f'{prim}, Tot. Fits = {totEvents}, Bad Fits = {totBadFits}')

            if iprim == 0:            
                ax.set_xticks(xVals)
                ax.set_xticklabels(xLabels, minor=False, rotation=45)

        ax.legend(loc="best", prop={"size": 12})

        ax.tick_params(direction="in", which="both", axis="both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")

        ax.text(0.33, 0.71, rf"$\theta_{{\rm zen}} = {self.minDeg}^{{\circ}}-{self.maxDeg}^{{\circ}}$", transform=ax.transAxes, fontsize=14)
        ax.text(0.33, 0.65, rf"lg(E) = {self.minLgE}$-${self.maxLgE}", transform=ax.transAxes, fontsize=14)

        if self.flagDataCuts == True:
            fileToSave = path + "/plots/misc/" + self.observatoryName + "_NumberBadFitTypes_" + fittype + f"_zen{self.minDeg}_{self.maxDeg}_lgE{self.minLgE}_{self.maxLgE}_DataCutApplied.pdf"
        else:
            fileToSave = path + "/plots/misc/" + self.observatoryName + "_NumberBadFitTypes_" + fittype + f"_zen{self.minDeg}_{self.maxDeg}_lgE{self.minLgE}_{self.maxLgE}.pdf"
        fig.savefig(fileToSave, bbox_inches="tight")
        print("Saved", fileToSave)


    def GetNumberOfCutEvents(self):

        for iprim, prim in enumerate(self.data.keys()):

            totalEvents = 0
            eventsToBeCut = 0

            prevZen = 0
            prevAzi = 0

            for event in self.eventList:

                name = self.primaryNames[str(event.primary)]

                # Only count number of bad fits for the primary in question from function call
                if name != prim:
                    continue

                totalEvents += 1

                # Apply cut for xmax because very large xmax values are unphysical
                if not 0 < event.xmax < 1500:
                    eventsToBeCut += 1
                    continue

                # Apply data cuts if keyword provided... (maybe put in GetValues instead?)
                if (self.flagDataCuts == True) and (self.flagGHFits == True):
                    if event.sigmaXmaxfit == np.inf or event.sigmaRfit == np.inf or event.sigmaLfit == np.inf:
                        eventsToBeCut += 1
                        continue
                    elif event.sigmaXmaxfit > 5.0 or event.sigmaRfit > 0.05 or event.sigmaLfit > 5.0:
                        eventsToBeCut += 1
                        continue
                    elif event.Xmaxfit < 0.0:
                        eventsToBeCut += 1
                        continue
                elif (self.flagDataCuts == True) and (self.flagGHFits == False):
                    if event.sigmaXmaxfitAndringa == np.inf or event.sigmaRfitAndringa == np.inf or event.sigmaLfitAndringa == np.inf:
                        eventsToBeCut += 1
                        continue
                    elif event.sigmaXmaxfitAndringa > 5.0 or event.sigmaRfitAndringa > 0.05 or event.sigmaLfitAndringa > 5.0:
                        eventsToBeCut += 1
                        continue
                    elif event.RfitAndringa < 0.0:
                        eventsToBeCut += 1
                        continue

                zen = event.zenith
                azi = event.azimuth

                # Do not repeat showers
                if zen == prevZen and azi == prevAzi:
                    eventsToBeCut += 1
                    continue

            print(f"Total Number of Events for {prim}: {totalEvents}")
            print(f"Events to be Cut From Analysis for {prim}: {eventsToBeCut}")


    def GetNumberOfEventsWhereCORSIKAFitFails(self, priorToDataCuts=False):
        # Consider a "failed" CORSIKA fit as one that is different from the python fit by >= 10 g/cm^2

        for iprim, prim in enumerate(self.data.keys()):

            totalEvents = 0
            eventsCORSIKAFail = 0
            eventsToBeCut = 0

            prevZen = 0
            prevAzi = 0

            for event in self.eventList:

                name = self.primaryNames[str(event.primary)]

                # Only count number of bad fits for the primary in question from function call
                if name != prim:
                    continue

                totalEvents += 1

                if priorToDataCuts == True:
                    if self.flagGHFits == True:
                        delta = abs(event.xmax - event.Xmaxfit)
                        if delta >= 10.0:
                            eventsCORSIKAFail += 1
                    elif self.flagGHFits == False:
                        delta = abs(event.xmax - event.XmaxfitAndringa)
                        if delta >= 10.0:
                            eventsCORSIKAFail += 1

                # Apply cut for xmax because very large xmax values are unphysical
                if not 0 < event.xmax < 1500:
                    eventsToBeCut += 1
                    continue

                # Apply data cuts if keyword provided... (maybe put in GetValues instead?)
                if (self.flagDataCuts == True) and (self.flagGHFits == True):
                    if event.sigmaXmaxfit == np.inf or event.sigmaRfit == np.inf or event.sigmaLfit == np.inf:
                        eventsToBeCut += 1
                        continue
                    elif event.sigmaXmaxfit > 5.0 or event.sigmaRfit > 0.05 or event.sigmaLfit > 5.0:
                        eventsToBeCut += 1
                        continue
                    elif event.Xmaxfit < 0.0:
                        eventsToBeCut += 1
                        continue
                elif (self.flagDataCuts == True) and (self.flagGHFits == False):
                    if event.sigmaXmaxfitAndringa == np.inf or event.sigmaRfitAndringa == np.inf or event.sigmaLfitAndringa == np.inf:
                        eventsToBeCut += 1
                        continue
                    elif event.sigmaXmaxfitAndringa > 5.0 or event.sigmaRfitAndringa > 0.05 or event.sigmaLfitAndringa > 5.0:
                        eventsToBeCut += 1
                        continue
                    elif event.RfitAndringa < 0.0:
                        eventsToBeCut += 1
                        continue

                zen = event.zenith
                azi = event.azimuth

                # Do not repeat showers
                if zen == prevZen and azi == prevAzi:
                    eventsToBeCut += 1
                    continue

                if priorToDataCuts == False:
                    if self.flagGHFits == True:
                        delta = abs(event.xmax - event.Xmaxfit)
                        if delta >= 10.0:
                            eventsCORSIKAFail += 1
                    elif self.flagGHFits == False:
                        delta = abs(event.xmax - event.XmaxfitAndringa)
                        if delta >= 10.0:
                            eventsCORSIKAFail += 1

            print(f"Total Number of Events for {prim}: {totalEvents}")

            if priorToDataCuts == True:
                print(f"Events With Failed CORSIKA Fit (Before Data Cuts) for {prim}: {eventsCORSIKAFail}")
                print(f"Events to be Cut From Analysis (including failed CORSIKA fits) for {prim}: {eventsToBeCut}")
            else:
                print(f"Events With Failed CORSIKA Fit (After Data Cuts) for {prim}: {eventsCORSIKAFail}")
                print(f"Events to be Cut From Analysis (NOT including failed CORSIKA fits) for {prim}: {eventsToBeCut}")


