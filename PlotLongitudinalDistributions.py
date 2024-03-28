#!/usr/bin/env python3

######################################
# Description of file and file usage #
######################################

# File to fit the longitudinal distributions from CORSIKA .long files ro determine R and L parameters

# Run by executing the command...
# ./PlotLongitudinalDistributions.py PATH_TO_LONG_FILE --kwargs

# PATH_TO_ASCII_FILES for diffrent experiments:
# IceCube -> /home/acoleman/sim/coreas/data/continuous/star-pattern/PRIMARY/ENERGY_BIN/ZENITH_BIN/xxxxxx/DATxxxxxx.long
# Auger   -> Note currently accesible unless saved by hand

# Note: Fits will nominally be to all particle longitudinal distributions but can be performed for only electromagentic (e+/e-) particles

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

rc("font", size=18.0)
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "dejavuserif"

ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, nargs="?", default="DATxxxxxx.long", help="Path to .long CORSIKA output file")
parser.add_argument("--observatory", type=str, nargs="?", required=True, default="IceCube", help="Name of observatory (either IceCube or Auger)")
#parser.add_argument("--zenithRange", type=float, nargs=2, default=[40.0, 50.0], help="Zenith range of data to plot")
#parser.add_argument("--energyRange", type=float, nargs=2, default=[16.5, 16.9], help="lg(E) energy range of data to plot")
parser.add_argument("--emParticles", action="store_true", help="If set, will plot longitudinal distributions from the electromagnetic particles only.")
parser.add_argument("--applyFits", action="store_true", help="If set, will apply GH fits to the longitudinal distributions.")
parser.add_argument("--compareCONEX", action="store_true", help="If applicable, will plot the original CORSIKA .long file to the CONEX .long file.")
parser.add_argument("--useEnergyDeposit", action="store_true", help="If set, will plot/fit the energy deposit instead of the particle numbers.")
args = parser.parse_args()

if os.path.isfile(args.input) == False:
    raise ValueError("Input filename does not exist, try again.")

if args.observatory == "Auger":
    raise ValueError("Can not plot the longitudinal profiles for Auger unless the profiles have been saved individually.")

fileToRead = args.input

#fileToRead = "/home/acoleman/sim/coreas/data/continuous/star-pattern/proton/lgE_17.0/sin2_0.5/000001/DAT000001.long"

#fileToRead = "/home/acoleman/sim/coreas/data/continuous/star-pattern/proton/lgE_18.1/sin2_0.0/000000/DAT000000.long" # Originally good profile fit
#fileToRead = "/home/acoleman/sim/coreas/data/continuous/star-pattern/proton/lgE_18.1/sin2_0.0/000001/DAT000001.long" # Originally bad profile fit

#fileToRead = "/home/acoleman/sim/coreas/data/continuous/star-pattern/proton/lgE_16.0/sin2_0.7/000027/DAT000027.long" # One out of three fits w/ Xmax < 0 (after data cuts)
#fileToRead = "/home/acoleman/sim/coreas/data/continuous/star-pattern/proton/lgE_17.3/sin2_0.0/000041/DAT000041.long" # Two out of three fits w/ Xmax < 0 (after data cuts)
#fileToRead = "/home/acoleman/sim/coreas/data/continuous/star-pattern/proton/lgE_17.7/sin2_0.1/000138/DAT000138.long" # Three out of three fits w/ Xmax < 0 (after data cuts)
#fileToRead = "/home/acoleman/sim/coreas/data/continuous/star-pattern/proton/lgE_16.0/sin2_0.6/000118/DAT000118.long" # Four out of five fits w/ Xmax < 0 (before data cuts)
#fileToRead = "/home/acoleman/sim/coreas/data/continuous/star-pattern/proton/lgE_16.9/sin2_0.3/000008/DAT000008.long" # Five out of five fits w/ Xmax < 0 (before data cuts)


fileSplit = fileToRead.split("/")
fileExtension = fileSplit[-1].rsplit(".")
if fileExtension[-1] != "long":
    raise ValueError("Input is not a valid file containing longitudinal profile information.")

prim = fileSplit[8]
engBin = fileSplit[9].split("_")[1]
zenBin = fileSplit[10].split("_")[1]
eventNum = fileSplit[11]

if args.compareCONEX == True:
    corsikaPath = fileToRead.rsplit("/", 1)
    additionalLongFile = corsikaPath[0] + "/backuplong.txt"
    if os.path.isfile(additionalLongFile) == False:
        raise ValueError("Can not compare CONEX to CORSIKA shower profile b/c this shower was not resimulated with CONEX!")

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
    if args.useEnergyDeposit == True:
        energyDeposit = 1
    else:
        energyDeposit = 0

    for iline, line in enumerate(file):
        if iline == 0:
            nLines = line.split()[3]
            continue

        if iline <= 2:
            continue  # Skip header

        cols = line.split()

        if "ENERGY" in cols and "DEPOSIT" in cols:
            if args.useEnergyDeposit == True:
                energyDeposit -= 1
            else:
                energyDeposit += 1

        if "PARAMETERS" in cols:
            xmax = float(cols[4])
            x0 = float(cols[3])
            lambdaApprox = float(cols[5]) # Drop 1st + 2nd order corrections

            x0Prime = x0 - xmax
            Rcorsika = float(np.sqrt(lambdaApprox / abs(x0Prime))) # Shape parameter
            Lcorsika = float(np.sqrt(abs(x0Prime * lambdaApprox))) # Characteristic width

        if len(cols) != 10 or "FIT" in cols or "DEPTH" in cols:
            continue

        if energyDeposit > 0:
            continue  # Skip all unwanted lines in .long file (either particle numbers or energy deposit, depending on keywords)

        depths.append(float(cols[0]))
        positrons.append(float(cols[2]))
        electrons.append(float(cols[3]))
        muPlus.append(float(cols[4]))
        muMinus.append(float(cols[5]))
        if args.useEnergyDeposit == True:
            chargedParticles.append(float(cols[9]))
        else:
            chargedParticles.append(float(cols[7]))

    file.close()

    return depths, positrons, electrons, muPlus, muMinus, chargedParticles, xmax, Rcorsika, Lcorsika, x0, lambdaApprox


def FindGroundIndex(ground, depths):
    absDepthToGround = []

    for i in range(len(depths)):
        absDepthToGround.append(abs(depths[i] - ground))

    minDepthToGround = min(absDepthToGround)
    indGround = absDepthToGround.index(minDepthToGround)

    return indGround

def AndringaFunction(X, Xmax, R, L):
    return (1 + (R * (X - Xmax) / L)) ** (1 / (R**2)) * np.exp(-1. * (X - Xmax) / (L * R))


def GHFunction(X, Nmax, Xmax, X0, lamb):
    return Nmax * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / lamb) * np.exp((Xmax - X) / lamb)


def FitLongitudinalProfile(depths, chargedParticles, NmaxGuess, XmaxGuess, X0Guess, lambGuess, calcChi2=False, shiftDepths=False):

    if shiftDepths == True:
        depthArray = np.asarray(depths) + 100.0 # Shift all depths by 100 g/cm^2 to see if we get physical results...
    else:
        depthArray = np.asarray(depths) # Original

    particleArray = np.asarray(chargedParticles)

    # Use Poissonian counting statistics to estimate an uncertainty in the charged particle number
    # Take uncertainty as ratio of uncertainty in bin to particle number in bin so bins with more particles have less uncertainty than those with less particles
    uncerts = 1.0 / np.sqrt(particleArray)

    # Insert a try-except statement for poor fits...
    try:
        if shiftDepths == True:
            popt, pcov = curve_fit(GHFunction, depthArray, particleArray, p0=[NmaxGuess, XmaxGuess+100.0, X0Guess, lambGuess], sigma=uncerts)
        else:
            popt, pcov = curve_fit(GHFunction, depthArray, particleArray, p0=[NmaxGuess, XmaxGuess, X0Guess, lambGuess], sigma=uncerts)
    except RuntimeError:
        print("Could not fit the profile to this function type!")
        if calcChi2 == True:
            return np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf
        else:
            return np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf
       

    perr = np.sqrt(np.diag(pcov))

    NmaxFit = popt[0]
    XmaxFit = popt[1]
    X0Fit = popt[2]
    lambFit = popt[3]

    NmaxSigma = perr[0]
    XmaxSigma = perr[1]
    X0Sigma = perr[2]
    lambSigma = perr[3]

    #print(NmaxFit, XmaxFit, X0Fit, lambFit)
    #print(NmaxSigma, XmaxSigma, X0Sigma, lambSigma)

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

    if shiftDepths == True:
        XmaxFit = popt[1] - 100.0 # Shift Xmax from fit back to real xmax
        X0Fit = popt[2] - 100.0 # Shift X0 value from fit back to true X0 for observed profile
    else:
        XmaxFit = popt[1]
        X0Fit = popt[2]

    if calcChi2 == True:
        fit = GHFunction(depthArray, *popt)
        resid = particleArray - fit
        #chisq = np.sum((resid / uncerts)**2)
        chisq = np.sum((resid / np.sqrt(particleArray))**2)        
        df = len(depthArray) - 4
        reducedChisq = chisq / df

        return RFit, RFitSigma, LFit, LFitSigma, XmaxFit, XmaxSigma, NmaxFit, X0Fit, lambFit, NmaxSigma, X0Sigma, lambSigma, chisq, reducedChisq
    else:
        return RFit, RFitSigma, LFit, LFitSigma, XmaxFit, XmaxSigma, NmaxFit, X0Fit, lambFit, NmaxSigma, X0Sigma, lambSigma


def FitLongitudinalProfileAndringa(depths, NprimeArray, XmaxGuess, RGuess, LGuess, calcChi2=False, shiftDepths=False):

    if shiftDepths == True:
        depthArray = np.asarray(depths) + 100.0 # Shift all depths by 100 g/cm^2 to see if we get physical results...
    else:
        depthArray = np.asarray(depths) # Original

    # Use Poissonian counting statistics to estimate an uncertainty in the charged particle number
    # Take uncertainty as ratio of uncertainty in bin to particle number in bin so bins with more particles have less uncertainty than those with less particles
    uncerts = 1.0 / np.sqrt(NprimeArray)

    # Insert a try-except statement for poor fits...
    try:
        if shiftDepths == True:
            popt, pcov = curve_fit(AndringaFunction, depthArray, NprimeArray, p0=[XmaxGuess+100.0, RGuess, LGuess], sigma=uncerts)
        else:
            popt, pcov = curve_fit(AndringaFunction, depthArray, NprimeArray, p0=[XmaxGuess, RGuess, LGuess], sigma=uncerts)
    except RuntimeError:
        print("Could not fit the profile to this function type!")
        if calcChi2 == True:
            return np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf
        else:
            return np.inf, np.inf, np.inf, np.inf, np.inf, np.inf


    perr = np.sqrt(np.diag(pcov))

    XmaxAndringaFit = popt[0]
    RAndringaFit = popt[1]
    LAndringaFit = popt[2]

    XmaxAndringaSigma = perr[0]
    RAndringaSigma = perr[1]
    LAndringaSigma = perr[2]

    if shiftDepths == True:
        XmaxAndringaFit = popt[0] - 100.0 # Shift Xmax from fit back to real xmax
    else:
        XmaxAndringaFit = popt[0]

    if calcChi2 == True:
        fit = AndringaFunction(depthArray, *popt)
        resid = NprimeArray - fit
        #chisq = np.sum((resid / uncerts)**2)
        chisq = np.sum((resid / np.sqrt(NprimeArray))**2)
        df = len(depthArray) - 3
        reducedChisq = chisq / df

        return XmaxAndringaFit, XmaxAndringaSigma, RAndringaFit, RAndringaSigma, LAndringaFit, LAndringaSigma, chisq, reducedChisq
    else:
        return XmaxAndringaFit, XmaxAndringaSigma, RAndringaFit, RAndringaSigma, LAndringaFit, LAndringaSigma


def remove_zeros(listToUpdate, pairedList):
    for i in reversed(range(len(listToUpdate))):
        if listToUpdate[i] == 0:
            del listToUpdate[i]
            del pairedList[i]
    return listToUpdate, pairedList


depths, positrons, electrons, muPlus, muMinus, chargedParticles, xmax, Rcorsika, Lcorsika, x0, lambdaApprox = LongFileParser(fileToRead)

totalEM = np.array(positrons) + np.array(electrons)
ixmax = np.argmax(totalEM)

if xmax > 1700:  # Sometimes corsika fits of the profile fail
    xmax = depths[ixmax]
    Rcorsika = -1
    Lcorsika = -1

if args.emParticles == True:
    Nmax = np.max(totalEM)
else:
    Nmax = np.max(chargedParticles)
NmaxGuess = Nmax
XmaxGuess = depths[ixmax]
#X0Guess = x0
X0Guess = 0
#lambGuess = lambdaApprox
lambGuess = 80.0

chargedParticles, depths = remove_zeros(chargedParticles, depths)

print(NmaxGuess, XmaxGuess, X0Guess, lambGuess)

# Remove the last point in the longitudinal distributions b/c it is not physical (sometimes below ground)
# Agnieszka suggested removing the final two points but start with only the final point at first
depths.pop()
positrons.pop()
electrons.pop()
muPlus.pop()
muMinus.pop()
chargedParticles.pop()
depths.pop()
positrons.pop()
electrons.pop()
muPlus.pop()
muMinus.pop()
chargedParticles.pop()

if args.emParticles == True:
    totalEM = np.array(positrons) + np.array(electrons)
    if args.compareCONEX == True:
        plotLabel = "EM Shower Profile (CONEX)"
    else:
        plotLabel = "Electromagnetic Shower Profile"
    RFit, RFitSigma, LFit, LFitSigma, XmaxFit, XmaxSigma, NmaxFit, X0Fit, lambFit, NmaxSigma, X0Sigma, lambSigma = FitLongitudinalProfile(depths, totalEM, NmaxGuess, XmaxGuess, X0Guess, lambGuess, calcChi2=False, shiftDepths=False)
else:
    if args.compareCONEX == True:
        plotLabel = "Shower Profile (CONEX)"
    else:
        plotLabel = "Shower Profile"
    RFit, RFitSigma, LFit, LFitSigma, XmaxFit, XmaxSigma, NmaxFit, X0Fit, lambFit, NmaxSigma, X0Sigma, lambSigma = FitLongitudinalProfile(depths, chargedParticles, NmaxGuess, XmaxGuess, X0Guess, lambGuess, calcChi2=False, shiftDepths=False)

#RFit, RFitSigma, LFit, LFitSigma, XmaxFit, XmaxSigma, NmaxFit, X0Fit, lambFit, NmaxSigma, X0Sigma, lambSigma, GHChisq, GHReducedChisq = FitLongitudinalProfile(depths, chargedParticles, NmaxGuess, XmaxGuess, X0Guess, lambGuess, calcChi2=True, shiftDepths=False)
#RFit, RFitSigma, LFit, LFitSigma, XmaxFit, XmaxSigma, NmaxFit, X0Fit, lambFit, NmaxSigma, X0Sigma, lambSigma = FitLongitudinalProfile(depths, chargedParticles, NmaxGuess, XmaxGuess, X0Guess, lambGuess, calcChi2=False, shiftDepths=False)

XVals = np.asarray(depths)
if args.emParticles == True:
    NVals = totalEM
else:
    NVals = np.asarray(chargedParticles)

fitCorsikaGH = GHFunction(XVals, Nmax, xmax, x0, lambdaApprox)
fitPythonGH = GHFunction(XVals, NmaxFit, XmaxFit, X0Fit, lambFit)

if args.compareCONEX == True:
    depthsOG, positronsOG, electronsOG, muPlusOG, muMinusOG, chargedParticlesOG, xmaxOG, RcorsikaOG, LcorsikaOG, x0OG, lambdaApproxOG = LongFileParser(additionalLongFile)

    totalEMOG = np.array(positronsOG) + np.array(electronsOG)
    ixmaxOG = np.argmax(totalEMOG)

    if args.emParticles == True:
        NmaxOG = np.max(totalEMOG)
    else:
        NmaxOG = np.max(chargedParticlesOG)
    NmaxGuessOG = NmaxOG
    XmaxGuessOG = depthsOG[ixmaxOG]
    X0GuessOG = 0
    lambGuessOG = 80.0

    chargedParticlesOG, depthsOG = remove_zeros(chargedParticlesOG, depthsOG)

    # Remove the last point in the longitudinal distributions b/c it is not physical (sometimes below ground)
    # Agnieszka suggested removing the final two points but start with only the final point at first
    depthsOG.pop()
    positronsOG.pop()
    electronsOG.pop()
    muPlusOG.pop()
    muMinusOG.pop()
    chargedParticlesOG.pop()
    depthsOG.pop()
    positronsOG.pop()
    electronsOG.pop()
    muPlusOG.pop()
    muMinusOG.pop()
    chargedParticlesOG.pop()

    if args.emParticles == True:
        totalEMOG = np.array(positronsOG) + np.array(electronsOG)
        plotLabelOG = "EM Shower Profile (CORSIKA)"
        RFitOG, RFitSigmaOG, LFitOG, LFitSigmaOG, XmaxFitOG, XmaxSigmaOG, NmaxFitOG, X0FitOG, lambFitOG, NmaxSigmaOG, X0SigmaOG, lambSigmaOG = FitLongitudinalProfile(depthsOG, totalEMOG, NmaxGuessOG, XmaxGuessOG, X0GuessOG, lambGuessOG, calcChi2=False, shiftDepths=False)
    else:
        plotLabelOG = "Shower Profile (CORSIKA)"
        RFitOG, RFitSigmaOG, LFitOG, LFitSigmaOG, XmaxFitOG, XmaxSigmaOG, NmaxFitOG, X0FitOG, lambFitOG, NmaxSigmaOG, X0SigmaOG, lambSigmaOG = FitLongitudinalProfile(depthsOG, chargedParticlesOG, NmaxGuessOG, XmaxGuessOG, X0GuessOG, lambGuessOG, calcChi2=False, shiftDepths=False)

    XValsOG = np.asarray(depthsOG)
    if args.emParticles == True:
        NValsOG = totalEMOG
    else:
        NValsOG = np.asarray(chargedParticlesOG)

    fitCorsikaGHOG = GHFunction(XValsOG, NmaxOG, xmaxOG, x0OG, lambdaApproxOG)
    fitPythonGHOG = GHFunction(XValsOG, NmaxFitOG, XmaxFitOG, X0FitOG, lambFitOG)



nRows = 1
nCols = 2
#nCols = 1

gs = gridspec.GridSpec(nRows, nCols, wspace=0.3, hspace=0.3)
fig = plt.figure(figsize=(nCols * 18.0 / 2.54, nRows * 15.0 / 2.54))

ax = fig.add_subplot(gs[0])

ax.scatter(XVals, NVals, color='blue', marker='o', label=plotLabel)
if args.applyFits == True:
    #ax.plot(XVals, fitCorsikaGH, color='black', linestyle='-',
    #    label=fr'.long Fit' + '\n' + fr'$X_{{\rm max}}$ = {xmax:.2f}, $X_{{0}}$ = {x0:.2f},' + '\n' + fr'$\lambda$ = {lambdaApprox:.2f}, $N_{{\rm max}}$ = {Nmax:.2e}')
    if args.compareCONEX == True:
        ax.plot(XVals, fitPythonGH, color='red', linestyle='-',
            label=fr'scipy.curve_fit (CONEX)' + '\n' + fr'$X_{{\rm max}}$ = {XmaxFit:.2f}, $X_{{0}}$ = {X0Fit:.2f},' + '\n' + fr'$\lambda$ = {lambFit:.2f}, $N_{{\rm max}}$ = {NmaxFit:.2e}')
    else:
        ax.plot(XVals, fitPythonGH, color='red', linestyle='-',
            label=fr'scipy.curve_fit (CORSIKA)' + '\n' + fr'$X_{{\rm max}}$ = {XmaxFit:.2f}, $X_{{0}}$ = {X0Fit:.2f},' + '\n' + fr'$\lambda$ = {lambFit:.2f}, $N_{{\rm max}}$ = {NmaxFit:.2e}')


if args.compareCONEX == True:
    ax.scatter(XValsOG, NValsOG, color='green', marker='P', label=plotLabelOG)
    if args.applyFits == True:
        #ax.plot(XValsOG, fitCorsikaGHOG, color='black', linestyle=':',
        #    label=fr'CORSIKA Fit' + '\n' + fr'$X_{{\rm max}}$ = {xmaxOG:.2f}, $X_{{0}}$ = {x0OG:.2f},' + '\n' + fr'$\lambda$ = {lambdaApproxOG:.2f}, $N_{{\rm max}}$ = {NmaxOG:.2e}')
        ax.plot(XValsOG, fitPythonGHOG, color='red', linestyle=':',
            label=fr'scipy.curve_fit (CORSIKA)' + '\n' + fr'$X_{{\rm max}}$ = {XmaxFitOG:.2f}, $X_{{0}}$ = {X0FitOG:.2f},' + '\n' + fr'$\lambda$ = {lambFitOG:.2f}, $N_{{\rm max}}$ = {NmaxFitOG:.2e}')

ax.set_xlabel(r"X [g cm$^{-2}$]")
if args.emParticles == True:
    ax.set_ylabel(r"N (total e$^{+/-}$)")
else:
    ax.set_ylabel(r"N (charged particles)")
ax.legend(loc="best", fontsize=10, ncol=1)
#ax.set_title("Gaisser-Hillas Function")
#ax.text(0.2, 0.45, rf"GH: $\chi^{{2}}$ = {GHChisq:.2e}, Red. $\chi^{{2}}$ = {GHReducedChisq:.2e}", transform=ax.transAxes, fontsize=10)


XPrimeValsCORSIKA = np.asarray(depths) - xmax
if args.emParticles == True:
    NPrimeValsCORSIKA = totalEM / Nmax
else:
    NPrimeValsCORSIKA = np.asarray(chargedParticles) / Nmax

XPrimeVals = np.asarray(depths) - XmaxFit
#NPrimeVals = np.asarray(chargedParticles) / NmaxFit


fitCorsika = AndringaFunction(XVals, xmax, Rcorsika, Lcorsika)
fitPython = AndringaFunction(XVals, XmaxFit, RFit, LFit)


# Do some testing here of different parameter guesses...
#RGuess = Rcorsika
RGuess = np.sqrt(lambGuess / abs(X0Guess - XmaxGuess))
#LGuess = Lcorsika
LGuess = np.sqrt(abs(X0Guess - XmaxGuess) * lambGuess)

print(RGuess, LGuess)

#XmaxAndringaFit, XmaxAndringaSigma, RAndringaFit, RAndringaSigma, LAndringaFit, LAndringaSigma, AndringaChisq, AndringaReducedChisq = FitLongitudinalProfileAndringa(depths, NPrimeValsCORSIKA, XmaxGuess, RGuess, LGuess, calcChi2=True, shiftDepths=False)
XmaxAndringaFit, XmaxAndringaSigma, RAndringaFit, RAndringaSigma, LAndringaFit, LAndringaSigma = FitLongitudinalProfileAndringa(depths, NPrimeValsCORSIKA, XmaxGuess, RGuess, LGuess, calcChi2=False, shiftDepths=False)

XPrimeValsAndringa = np.asarray(depths) - XmaxAndringaFit

fitPythonAndringa = AndringaFunction(XVals, XmaxAndringaFit, RAndringaFit, LAndringaFit)


if args.compareCONEX == True:
    XPrimeValsCORSIKAOG = np.asarray(depthsOG) - xmaxOG
    if args.emParticles == True:
        NPrimeValsCORSIKAOG = totalEMOG / NmaxOG
    else:
        NPrimeValsCORSIKAOG = np.asarray(chargedParticlesOG) / NmaxOG

    XPrimeValsOG = np.asarray(depthsOG) - XmaxFitOG

    fitCorsikaOG = AndringaFunction(XValsOG, xmaxOG, RcorsikaOG, LcorsikaOG)
    fitPythonOG = AndringaFunction(XValsOG, XmaxFitOG, RFitOG, LFitOG)


    RGuessOG = np.sqrt(lambGuessOG / abs(X0GuessOG - XmaxGuessOG))
    LGuessOG = np.sqrt(abs(X0GuessOG - XmaxGuessOG) * lambGuessOG)


    XmaxAndringaFitOG, XmaxAndringaSigmaOG, RAndringaFitOG, RAndringaSigmaOG, LAndringaFitOG, LAndringaSigmaOG = FitLongitudinalProfileAndringa(depthsOG, NPrimeValsCORSIKAOG, XmaxGuessOG, RGuessOG, LGuessOG, calcChi2=False, shiftDepths=False)

    XPrimeValsAndringaOG = np.asarray(depthsOG) - XmaxAndringaFitOG

    fitPythonAndringaOG = AndringaFunction(XValsOG, XmaxAndringaFitOG, RAndringaFitOG, LAndringaFitOG)


#nRows = 1
#nCols = 1

#gs = gridspec.GridSpec(nRows, nCols, wspace=0.3, hspace=0.3)
#fig = plt.figure(figsize=(nCols * 18.0 / 2.54, nRows * 15.0 / 2.54))

ax2 = fig.add_subplot(gs[1])

ax2.scatter(XPrimeValsCORSIKA, NPrimeValsCORSIKA, color='blue', marker='o', label=plotLabel)
if args.applyFits == True:
    #ax2.plot(XPrimeValsCORSIKA, fitCorsika, color='black', linestyle='-',
    #    label=fr'.long Fit' + '\n' + fr'R = {Rcorsika:.2f}, L = {Lcorsika:.2f},' + '\n' + fr'$X_{{\rm max}}$ = {xmax:.2f}')
    if args.compareCONEX == True:
        #ax2.plot(XPrimeVals, fitPython, color='red', linestyle='-',
        #    label=fr'scipy.curve_fit (GH CONEX)' + '\n' + fr'R = {RFit:.2f} $\pm$ {RFitSigma:.2f},' + '\n' + fr'L = {LFit:.2f} $\pm$ {LFitSigma:.2f}' + '\n' + fr'$X_{{\rm max}}$ = {XmaxFit:.2f} $\pm$ {XmaxSigma:.2f}')
        ax2.plot(XPrimeValsAndringa, fitPythonAndringa, color='cyan', linestyle='-',
            label=fr'scipy.curve_fit (Param. CONEX)' + '\n' + fr'R = {RAndringaFit:.2f} $\pm$ {RAndringaSigma:.2f},' + '\n' + fr'L = {LAndringaFit:.2f} $\pm$ {LAndringaSigma:.2f}' + '\n' + fr'$X_{{\rm max}}$ = {XmaxAndringaFit:.2f} $\pm$ {XmaxAndringaSigma:.2f}')
    else:
        #ax2.plot(XPrimeVals, fitPython, color='red', linestyle='-',
        #    label=fr'scipy.curve_fit (GH CORSIKA)' + '\n' + fr'R = {RFit:.2f} $\pm$ {RFitSigma:.2f},' + '\n' + fr'L = {LFit:.2f} $\pm$ {LFitSigma:.2f}' + '\n' + fr'$X_{{\rm max}}$ = {XmaxFit:.2f} $\pm$ {XmaxSigma:.2f}')
        ax2.plot(XPrimeValsAndringa, fitPythonAndringa, color='cyan', linestyle='-',
            label=fr'scipy.curve_fit (Param. CORSIKA)' + '\n' + fr'R = {RAndringaFit:.2f} $\pm$ {RAndringaSigma:.2f},' + '\n' + fr'L = {LAndringaFit:.2f} $\pm$ {LAndringaSigma:.2f}' + '\n' + fr'$X_{{\rm max}}$ = {XmaxAndringaFit:.2f} $\pm$ {XmaxAndringaSigma:.2f}')

if args.compareCONEX == True:
    ax2.scatter(XPrimeValsCORSIKAOG, NPrimeValsCORSIKAOG, color='green', marker='P', label=plotLabelOG)
    if args.applyFits == True:
        #ax2.plot(XPrimeValsCORSIKAOG, fitCorsikaOG, color='black', linestyle=':',
        #    label=fr'CORSIKA Param. Fit' + '\n' + fr'R = {RcorsikaOG:.2f}, L = {LcorsikaOG:.2f},' + '\n' + fr'$X_{{\rm max}}$ = {xmaxOG:.2f}')
        ax2.plot(XPrimeValsAndringaOG, fitPythonAndringaOG, color='cyan', linestyle=':',
            label=fr'scipy.curve_fit (Param. CORSIKA)' + '\n' + fr'R = {RAndringaFitOG:.2f} $\pm$ {RAndringaSigmaOG:.2f},' + '\n' + fr'L = {LAndringaFitOG:.2f} $\pm$ {LAndringaSigmaOG:.2f}' + '\n' + fr'$X_{{\rm max}}$ = {XmaxAndringaFitOG:.2f} $\pm$ {XmaxAndringaSigmaOG:.2f}')

ax2.set_xlabel(r"X$^{\prime}$ = X - X$_{\rm max}$ [g cm$^{-2}$]")
ax2.set_ylabel(r"N$^{\prime}$ = N/N$_{\rm max}$")
ax2.legend(loc="best", fontsize=10, ncol=1)
#ax2.set_title("Andringa Function")
#ax2.text(0.2, 0.55, rf"Andringa: $\chi^{{2}}$ = {AndringaChisq:.2e}, Red. $\chi^{{2}}$ = {AndringaReducedChisq:.2e}", transform=ax2.transAxes, fontsize=10)


if args.emParticles == True:
    fileInfo = "_OnlyEMParticles"
else:
    fileInfo = ""

if args.compareCONEX == True:
    fileCompare = "CompareCONEX-"
else:
    fileCompare = ""

if args.applyFits == True:
    filename = ABS_PATH_HERE + "/plots/misc/ShowerProfiles/Profile_wFits-" + fileCompare + prim + "-lgE_" + engBin + "-sin2_" + zenBin + "-" + eventNum + fileInfo + ".pdf"
else:
    filename = ABS_PATH_HERE + "/plots/misc/ShowerProfiles/OnlyProfile-" + fileCompare + prim + "-lgE_" + engBin + "-sin2_" + zenBin + "-" + eventNum + fileInfo + ".pdf"

plt.savefig(filename, bbox_inches="tight")
print("Saved", filename)

print("\n")

print(f"R (corsika): {Rcorsika:.3f}")
print(f"R Fit Value: {RFit:.3f} +/- {RFitSigma:.3f}")
print(f"R Fit Value (Andringa Fit): {RAndringaFit:.3f} +/- {RAndringaSigma:.3f}")
print(f"L (corsika): {Lcorsika:.3f}")
print(f"L Fit Value: {LFit:.3f} +/- {LFitSigma:.3f}")
print(f"L Fit Value (Andringa Fit): {LAndringaFit:.3f} +/- {LAndringaSigma:.3f}\n")

print(f"Xmax (corsika): {xmax:.3f}")
print(f"Xmax Fit Value: {XmaxFit:.3f} +/- {XmaxSigma:.3f}")
print(f"Xmax Fit Value (Andringa Fit): {XmaxAndringaFit:.3f} +/- {XmaxAndringaSigma:.3f}\n")
print(f"X0 (corsika): {x0:.3f}")
print(f"X0 Fit Value: {X0Fit:.3f} +/- {X0Sigma:.3f}")
print(f"Lambda (corsika): {lambdaApprox:.3f}")
print(f"Lambda Fit Value: {lambFit:.3f} +/- {lambSigma:.3f}")
print(f"Nmax (corsika): {Nmax:.3e}")
print(f"Nmax Fit Value: {NmaxFit:.3e} +/- {NmaxSigma:.3e}")

#print(f"GH Chi-Sq.: {GHChisq:.3f}, GH Red. Chi-Sq.: {GHReducedChisq:.3f}")
#print(f"And. Chi-Sq.: {AndringaChisq:.3f}, And. Red. Chi-Sq.: {AndringaReducedChisq:.3f}")
