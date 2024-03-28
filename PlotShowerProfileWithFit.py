#!/usr/bin/env python3

######################################
# Description of file and file usage #
######################################

# File to fit the longitudinal distributions from CORSIKA .long files to determine R and L parameters
# Fits will be to all charged particle or electron/positron number longitudinal distributions

# Run as:
# ./PlotShowerWithFit.py PATH_TO_LONG_FILE --kwargs

# Notes:
# I have been wanting to include the fit to the energy deposit as a keyword instead of just the particle number
# but have been too busy with other things. I am not sure if this would work as all GH parameterizations have
# only been to the particle numbers...

######################
# End of description #
######################

import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import rc, rcParams
import argparse

rc("font", size=18.0)
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "dejavuserif"

ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, default="", help="Full path to CORSIKA .long output file")
parser.add_argument("--removeLastXDataPoints", type=int, default=0, help="Removes the final X points from the fit, where X is supplied by the user")
parser.add_argument("--electronDistribution", action="store_true", help="If set, will plot/fit the electron/positron shower profile")
parser.add_argument("--GHFit", action="store_true", help="If set, will plot/fit a GH fit (non-parameterized) to the data. Fit done with curve_fit")
args = parser.parse_args()


def LongFileParser(filename):
    depths = []
    positrons = []
    electrons = []
    muPlus = []
    muMinus = []
    chargedParticles = []

    file = open(filename, "r")
    nLines = 0
    xmax = -1
    Rcorsika = -1
    Lcorsika = -1
    energyDeposit = 0
    for iline, line in enumerate(file):
        if iline == 0:
            nLines = line.split()[3]
            continue

        if iline <= 2:
            continue  # Skip header

        cols = line.split()

        if "ENERGY" in cols and "DEPOSIT" in cols:
            energyDeposit += 1

        if "PARAMETERS" in cols:
            xmax = float(cols[4])
            x0 = float(cols[3])
            lambdaApprox = float(cols[5]) # Drop 1st + 2nd order corrections (see CORSIKA user guide Section 4.65)

            x0Prime = x0 - xmax
            Rcorsika = float(np.sqrt(lambdaApprox / abs(x0Prime))) # Shape parameter
            Lcorsika = float(np.sqrt(abs(x0Prime * lambdaApprox))) # Characteristic width

        if len(cols) != 10 or "FIT" in cols:
            continue  # Skip lines of text

        if energyDeposit > 0:
            continue  # Skip all energy deposit lines in .long file

        depths.append(float(cols[0]))
        positrons.append(float(cols[2]))
        electrons.append(float(cols[3]))
        muPlus.append(float(cols[4]))
        muMinus.append(float(cols[5]))
        chargedParticles.append(float(cols[7]))

    file.close()

    return depths, positrons, electrons, muPlus, muMinus, chargedParticles, xmax, Rcorsika, Lcorsika, x0, lambdaApprox


def GHFunction(X, Nmax, Xmax, X0, lamb):
    return Nmax * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / lamb) * np.exp((Xmax - X) / lamb)


def ParamGHFunction(X, Xmax, R, L):  # see DOI: 10.1016/j.astropartphys.2010.10.002 for definition
    return (1 + (R * (X - Xmax) / L)) ** (1 / (R**2)) * np.exp(-1. * (X - Xmax) / (L * R))


def FitProfile(depths, chargedParticles, NmaxGuess, XmaxGuess, X0Guess, lambGuess, shift=False):
    if (shift == True):
        depthArray = np.asarray(depths) + 100.0  # Shift all depths by 100 g/cm^2 to (needed so python will not throw errors for positive X0 values...)
    else:
        depthArray = np.asarray(depths)

    particleArray = np.asarray(chargedParticles)

    # Use Poissonian counting statistics to estimate an uncertainty in the charged particle number
    # Take uncertainty as ratio of uncertainty in bin to particle number in bin so bins with more particles have less uncertainty than those with less particles
    uncerts = 1.0 / np.sqrt(particleArray)

    # Insert a try-except statement for poor fits...
    try:
        if (shift == True):
            popt, pcov = curve_fit(GHFunction, depthArray, particleArray, p0=[NmaxGuess, XmaxGuess+100.0, X0Guess, lambGuess], sigma=uncerts)
        else:
            popt, pcov = curve_fit(GHFunction, depthArray, particleArray, p0=[NmaxGuess, XmaxGuess, X0Guess, lambGuess], sigma=uncerts)
    except RuntimeError:
        return np.inf, np.inf, np.inf, np.inf, np.inf, np.inf

    perr = np.sqrt(np.diag(pcov))

    NmaxFit = popt[0]
    XmaxFit = popt[1]
    X0Fit = popt[2]
    lambFit = popt[3]

    NmaxSigma = perr[0]
    XmaxSigma = perr[1]
    X0Sigma = perr[2]
    lambSigma = perr[3]

    X0Prime = abs(X0Fit - XmaxFit)

    RFit = np.sqrt(lambFit / X0Prime)
    LFit = np.sqrt(X0Prime * lambFit)

    # Squared terms found from error prop.
    sigmaRterm1 = (lambSigma ** 2) / (4 * lambFit * X0Prime)
    sigmaRterm2 = (lambFit * (X0Sigma ** 2 + XmaxSigma ** 2)) / (4 * X0Prime ** 3)

    RFitSigma = np.sqrt(sigmaRterm1 + sigmaRterm2)

    # Squared terms found from error prop.
    sigmaLterm1 = (X0Prime * lambSigma ** 2) / (4 * lambFit)
    sigmaLterm2 = (lambFit * (X0Sigma ** 2 + XmaxSigma ** 2)) / (4 * X0Prime)

    LFitSigma = np.sqrt(sigmaLterm1 + sigmaLterm2)

    if (shift == True):
        XmaxFit = popt[1] - 100.0 # Shift Xmax from fit back to real xmax
        X0Fit = popt[2] - 100.0 # Shift X0 value from fit back to true X0 for observed profile    

    return RFit, RFitSigma, LFit, LFitSigma, XmaxFit, XmaxSigma


def FitProfileParameterized(depths, NprimeArray, XmaxGuess, RGuess, LGuess, shift=False):
    if (shift == True):
        depthArray = np.asarray(depths) + 100.0  # Shift all depths by 100 g/cm^2 (only for testing purposes)
    else:
        depthArray = np.asarray(depths)

    # Use Poissonian counting statistics to estimate an uncertainty in the charged particle number
    # Take uncertainty as ratio of uncertainty in bin to particle number in bin so bins with more particles have less uncertainty than those with less particles
    uncerts = 1.0 / np.sqrt(NprimeArray)

    # Insert a try-except statement for poor fits...
    try:
        if (shift == True):
            popt, pcov = curve_fit(ParamGHFunction, depthArray, NprimeArray, p0=[XmaxGuess+100.0, RGuess, LGuess], sigma=uncerts)
        else:
            popt, pcov = curve_fit(ParamGHFunction, depthArray, NprimeArray, p0=[XmaxGuess, RGuess, LGuess], sigma=uncerts)
    except RuntimeError:
        return np.inf, np.inf, np.inf, np.inf, np.inf, np.inf

    perr = np.sqrt(np.diag(pcov))

    XmaxAndringaFit = popt[0]
    RAndringaFit = popt[1]
    LAndringaFit = popt[2]

    XmaxAndringaSigma = perr[0]
    RAndringaSigma = perr[1]
    LAndringaSigma = perr[2]

    if (shift == True):
        XmaxAndringaFit = popt[0] - 100.0  # Shift Xmax from fit back to real xmax

    return XmaxAndringaFit, XmaxAndringaSigma, RAndringaFit, RAndringaSigma, LAndringaFit, LAndringaSigma


def remove_zeros(listToUpdate, pairedList):
    for i in reversed(range(len(listToUpdate))):
        if listToUpdate[i] == 0:
            del listToUpdate[i]
            del pairedList[i]
    return listToUpdate, pairedList


fileToRead = args.input
depths, positrons, electrons, muPlus, muMinus, chargedParticles, xmax, Rcorsika, Lcorsika, x0, lambdaApprox = LongFileParser(fileToRead)

rmExtension = fileToRead.rsplit(".", 1)[0]
simNumber = rmExtension.rsplit("/", 1)[-1]

# Remove final X points from plot/fit b/c they are not physical
# My understanding is some part of the shower front reaches ground which causes dip in particle numbers...
cutNum = args.removeLastXDataPoints
if cutNum > 0:
    del depths[-cutNum:]
    del positrons[-cutNum:]
    del electrons[-cutNum:]
    del muPlus[-cutNum:]
    del muMinus[-cutNum:]
    del chargedParticles[-cutNum:]


if args.electronDistribution:
    totalEM = np.array(positrons) + np.array(electrons)
    totalEMList = totalEM.tolist()
    totalEMList, depths = remove_zeros(totalEMList, depths)
    NvalsGH = totalEMList

    totalEM = np.array(totalEMList)
    ixmax = np.argmax(totalEM)
    Nmax = np.max(totalEM)
    NPrimeValsCORSIKA = totalEM / Nmax
else:
    chargedParticles, depths = remove_zeros(chargedParticles, depths)
    NvalsGH = chargedParticles
    
    chargedParticles = np.array(chargedParticles)
    ixmax = np.argmax(chargedParticles)
    Nmax = np.max(chargedParticles)
    NPrimeValsCORSIKA = chargedParticles / Nmax

# Define parameter guesses for fit
if ixmax > len(depths):
    XmaxGuess = depths[-1]
else:
    XmaxGuess = depths[ixmax]
NmaxGuess = Nmax
X0Guess = 0
lambGuess = 80.0

RGuess = np.sqrt(lambGuess / abs(X0Guess - XmaxGuess))
LGuess = np.sqrt(abs(X0Guess - XmaxGuess) * lambGuess)

XVals = np.asarray(depths)
XPrimeValsCORSIKA = np.asarray(depths) - xmax
fitCorsika = ParamGHFunction(XVals, xmax, Rcorsika, Lcorsika)

if args.GHFit:
    RFitShift, RFitSigmaShift, LFitShift, LFitSigmaShift, XmaxFitShift, XmaxSigmaShift = FitProfile(depths, NvalsGH, NmaxGuess, XmaxGuess, X0Guess, lambGuess, shift=True)
    print(xmax, Rcorsika, Lcorsika, RFitShift, RFitSigmaShift, LFitShift, LFitSigmaShift, XmaxFitShift, XmaxSigmaShift, end="\n")
    XPrimeVals = np.asarray(depths) - XmaxFitShift
    fitPython = ParamGHFunction(XVals, XmaxFitShift, RFitShift, LFitShift)
    plotString = "_GHFit"
else:
    XmaxAndringaFit, XmaxAndringaSigma, RAndringaFit, RAndringaSigma, LAndringaFit, LAndringaSigma = FitProfileParameterized(depths, NPrimeValsCORSIKA, XmaxGuess, RGuess, LGuess, shift=False)
    print(xmax, Rcorsika, Lcorsika, RAndringaFit, RAndringaSigma, LAndringaFit, LAndringaSigma, XmaxAndringaFit, XmaxAndringaSigma, end="\n")
    XPrimeValsAndringa = np.asarray(depths) - XmaxAndringaFit
    fitPythonAndringa = ParamGHFunction(XVals, XmaxAndringaFit, RAndringaFit, LAndringaFit)
    plotString = "_ParamGHFit"

nRows = 1
nCols = 1

gs = gridspec.GridSpec(nRows, nCols, wspace=0.3, hspace=0.3)
fig = plt.figure(figsize=(nCols * 18.0 / 2.54, nRows * 15.0 / 2.54))

ax = fig.add_subplot(gs[0])

ax.scatter(XPrimeValsCORSIKA, NPrimeValsCORSIKA, color='blue', label='Shower Profile')
ax.plot(XPrimeValsCORSIKA, fitCorsika, color='black', linestyle='--',
    label=fr'CORSIKA Fit' + '\n' + fr'R = {Rcorsika:.2f},' + '\n' + fr'L = {Lcorsika:.2f},' + '\n' + fr'$X_{{\rm max}}$ = {xmax:.2f}')
if args.GHFit:
    ax.plot(XPrimeVals, fitPython, color='red', linestyle='--',
        label=fr'scipy.curve_fit (GH Shifted)' + '\n' + fr'R = {RFitShift:.2f} $\pm$ {RFitSigmaShift:.2f},' + '\n' + fr'L = {LFitShift:.2f} $\pm$ {LFitSigmaShift:.2f},' + '\n' + fr'$X_{{\rm max}}$ = {XmaxFitShift:.2f} $\pm$ {XmaxSigmaShift:.2f}')
else:
    ax.plot(XPrimeValsAndringa, fitPythonAndringa, color='red', linestyle='--',
        label=fr'scipy.curve_fit (GH Param.)' + '\n' + fr'R = {RAndringaFit:.2f} $\pm$ {RAndringaSigma:.2f},' + '\n' + fr'L = {LAndringaFit:.2f} $\pm$ {LAndringaSigma:.2f},' + '\n' + fr'$X_{{\rm max}}$ = {XmaxAndringaFit:.2f} $\pm$ {XmaxAndringaSigma:.2f}')

ax.set_xlabel(r"X$^{\prime}$ = X - X$_{\rm max}$ [g cm$^{-2}$]")
if args.electronDistribution:
    ax.set_ylabel(r"N$^{\prime}$ = N/N$_{\rm max}$ (e$^{+/-}$ only)")
    distributionString = plotString + "_electronsOnly"
else:
    ax.set_ylabel(r"N$^{\prime}$ = N/N$_{\rm max}$ (total charged)")
    distributionString = plotString + "_totalChargedParticles"

ax.legend(loc="best", fontsize=10, ncol=1)
ax.set_title(f"CORSIKA Profile w/ Fit: {simNumber}")

filename = ABS_PATH_HERE + "/LongDistribution_wFit_" + simNumber + distributionString + ".pdf"
plt.savefig(filename, bbox_inches="tight")
print("Saved", filename)

print("\n")

print(f"R (corsika): {Rcorsika:.3f}")
print(f"L (corsika): {Lcorsika:.3f}")
print(f"Xmax (corsika): {xmax:.3f}\n")

if args.GHFit:
    print(f"R Fit Value (GH Fit): {RFitShift:.3f} +/- {RFitSigmaShift:.3f}")
    print(f"L Fit Value (GH Fit): {LFitShift:.3f} +/- {LFitSigmaShift:.3f}")
    print(f"Xmax Fit Value (GH Fit): {XmaxFitShift:.3f} +/- {XmaxSigmaShift:.3f}")
else:
    print(f"R Fit Value (GH Param. Fit): {RAndringaFit:.3f} +/- {RAndringaSigma:.3f}")
    print(f"L Fit Value (GH Param. Fit): {LAndringaFit:.3f} +/- {LAndringaSigma:.3f}")
    print(f"Xmax Fit Value (GH Param. Fit): {XmaxAndringaFit:.3f} +/- {XmaxAndringaSigma:.3f}")
