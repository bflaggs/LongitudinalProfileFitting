#!/usr/bin/env python3

######################################
# Description of file and file usage #
######################################

# File to plot the longitudinal distributions from CORSIKA .long files
# Will plot total charged particle distributions and all individual charged particle distributions
# Goal is to see which particle type contributes most when discrepant showers (ex. double bump) occur

# Run as:
# ./PlotShowerProfile_Updated.py PATH_TO_LONG_FILE --kwargs

# Notes:
# Could also include a vertical line statting where the ground is... would be defined as X_ground / np.cos(zenith_radians)
# To find X_ground (atmospheric depth at ground) will need to integrate equation:
# rho(h) = rho(0)*np.exp(-h/h0) from limits of -np.inf to h_ground (may have to swap limits, I can't remember...)
# Constants are rho(0)=0.001225g/cm^3 and h0~=8km
# h_ground comes from the steering file, so does the zenith angle

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
parser.add_argument("--excludePhotons", action="store_true", help="If set will exclude gammas and cherenkov photons from the plot")
parser.add_argument("--excludeElectrons", action="store_true", help="If set will exclude positrons/electrons from the plot and subtracts them from the charged particle profile")
parser.add_argument("--logYAxis", action="store_true", help="If set will plot the y-axis on a logarithmic scale")
parser.add_argument("--plotGround", action="store_true", help="If set will plot a vertical line representing the slant depth of the observation level")
args = parser.parse_args()

if args.excludePhotons == False and args.excludeElectrons == True:
    raise ValueError("No need at the moment to remove electrons but keep photons... Try again with different options")

def LongFileParserFull(filename):
    depths = []
    gammas = []
    positrons = []
    electrons = []
    muPlus = []
    muMinus = []
    hadrons = []
    chargedParticles = []
    nuclei = []
    cherenkov = []

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
        gammas.append(float(cols[1]))
        positrons.append(float(cols[2]))
        electrons.append(float(cols[3]))
        muPlus.append(float(cols[4]))
        muMinus.append(float(cols[5]))
        hadrons.append(float(cols[6]))
        chargedParticles.append(float(cols[7]))
        nuclei.append(float(cols[8]))
        cherenkov.append(float(cols[9]))

    file.close()

    return depths, gammas, positrons, electrons, muPlus, muMinus, hadrons, chargedParticles,  nuclei, cherenkov, xmax, Rcorsika, Lcorsika, x0, lambdaApprox


def FindGroundSlantDepth(filename, observatory="Auger"):
    if observatory == "IceCube":
        prefix = "SIM"
    elif observatory == "Auger":
        prefix = "RUN"
    else:
        raise ValueError("Not a valid observatory name. Try again using IceCube or Auger.")

    rmExtension = filename.rsplit(".", 1)
    simulation = rmExtension[0].rsplit("T", 1)
    steeringFile = prefix + simulation[-1] + ".inp"

    if not os.path.isfile(steeringFile):
        raise ValueError("Steering file does not exist in the current path...")

    file = open(steeringFile, "r")
    zenith = -1
    obslev = -1
    for iline, line in enumerate(file):
        cols = line.split()

        if "THETAP" in cols:
            zenith = float(cols[1])

        if "OBSLEV" in cols:
            obslev = float(cols[1])

    if zenith == -1 or obslev == -1:
        raise ValueError("Zenith angle or observation level not found in steering file.")

    file.close()

    obslevKM = obslev / 100000.  # convert observation level from cm to km
    zenithRAD = zenith / 180. * np.pi  # convert zenith from degrees to radians

    h0 = 8  # [km]
    rho0 = 0.001225  # [g/cm^3]

    # DO THE INTEGRATION HERE TO FIND ATMOSPHERIC DEPTH OF GROUND

    # Convert X_ground to slant depth
    #groundDepth = X_ground / np.cos(zenithRAD)

    return groundDepth


def GHFunction(X, Nmax, Xmax, X0, lamb):
    return Nmax * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / lamb) * np.exp((Xmax - X) / lamb)


def ParamGHFunction(X, Xmax, R, L):  # see DOI: 10.1016/j.astropartphys.2010.10.002 for definition
    return (1 + (R * (X - Xmax) / L)) ** (1 / (R**2)) * np.exp(-1. * (X - Xmax) / (L * R))


def remove_zeros(listToUpdate, pairedList):
    for i in reversed(range(len(listToUpdate))):
        if listToUpdate[i] == 0:
            del listToUpdate[i]
            del pairedList[i]
    return listToUpdate, pairedList


fileToRead = args.input
depths, gammas, positrons, electrons, muPlus, muMinus, hadrons, chargedParticles, nuclei, cherenkov, xmax, Rcorsika, Lcorsika, x0, lambdaApprox = LongFileParserFull(fileToRead)

# Remove final X points from plot/fit b/c they are not physical
# My understanding is some part of the shower front reaches ground which causes dip in particle numbers...
cutNum = args.removeLastXDataPoints
if cutNum > 0:
    del depths[-cutNum:]
    del gammas[-cutNum:]
    del positrons[-cutNum:]
    del electrons[-cutNum:]
    del muPlus[-cutNum:]
    del muMinus[-cutNum:]
    del hadrons[-cutNum:]
    del chargedParticles[-cutNum:]
    del nuclei[-cutNum:]
    del cherenkov[-cutNum:]

gammaProfile = np.array(gammas)
ePlusProfile = np.array(positrons)
eMinusProfile = np.array(electrons)
muPlusProfile = np.array(muPlus)
muMinusProfile = np.array(muMinus)
hadronProfile = np.array(hadrons)
chargedParticleProfile = np.array(chargedParticles)
nucleiProfile = np.array(nuclei)
cherenkovPhotonProfile = np.array(cherenkov)


XVals = np.asarray(depths)
#XPrimeValsCORSIKA = np.asarray(depths) - xmax


nRows = 1
nCols = 1

gs = gridspec.GridSpec(nRows, nCols, wspace=0.3, hspace=0.3)
fig = plt.figure(figsize=(nCols * 18.0 / 2.54, nRows * 15.0 / 2.54))

ax = fig.add_subplot(gs[0])

if args.excludePhotons == True and args.excludeElectrons == False:
    plotString = "_OnlyChargedParticles"
    ax.scatter(XVals, ePlusProfile, color='red', marker='^', label=r'e$^{+}$')
    ax.scatter(XVals, eMinusProfile, color='red', marker='*', label=r'e$^{-}$')
    ax.scatter(XVals, muPlusProfile, color='blue', marker='^', label=r'$\mu^{+}$')
    ax.scatter(XVals, muMinusProfile, color='blue', marker='*', label=r'$\mu^{-}$')
    ax.scatter(XVals, hadronProfile, color='cyan', marker='d', label='Hadrons')
    ax.scatter(XVals, chargedParticleProfile, color='black', marker='o', label='Charged Particles')
    ax.scatter(XVals, nucleiProfile, color='magenta', marker='p', label='Nuclei')
elif args.excludePhotons == True and args.excludeElectrons == True:
    plotString = "_NoPhotonsOrElectrons_NoNuclei"
    chargedParticleProfile = chargedParticleProfile - ePlusProfile - eMinusProfile
    ax.scatter(XVals, muPlusProfile, color='blue', marker='^', label=r'$\mu^{+}$')
    ax.scatter(XVals, muMinusProfile, color='blue', marker='*', label=r'$\mu^{-}$')
    ax.scatter(XVals, hadronProfile, color='cyan', marker='d', label='Hadrons')
    ax.scatter(XVals, chargedParticleProfile, color='black', marker='o', label=r'Charged Particles (no e$^{+/-}$)')
    #ax.scatter(XVals, nucleiProfile, color='magenta', marker='p', label='Nuclei')
else:
    plotString = ""
    ax.scatter(XVals, gammaProfile, color='green', marker='+', label=r'$\gamma$')
    ax.scatter(XVals, ePlusProfile, color='red', marker='^', label=r'e$^{+}$')
    ax.scatter(XVals, eMinusProfile, color='red', marker='*', label=r'e$^{-}$')
    ax.scatter(XVals, muPlusProfile, color='blue', marker='^', label=r'$\mu^{+}$')
    ax.scatter(XVals, muMinusProfile, color='blue', marker='*', label=r'$\mu^{-}$')
    ax.scatter(XVals, hadronProfile, color='cyan', marker='d', label='Hadrons')
    ax.scatter(XVals, chargedParticleProfile, color='black', marker='o', label='Charged Particles')
    ax.scatter(XVals, nucleiProfile, color='magenta', marker='p', label='Nuclei')
    ax.scatter(XVals, cherenkovPhotonProfile, color='green', marker='x', label='Cherenkov Photon')

ax.set_xlabel(r"X [g cm$^{-2}$]")
ax.set_ylabel("N")
#ax.set_title("Charged Particle Shower Profile w/ ")
ax.set_title("CORSIKA Profiles")

if args.logYAxis == True:
    ax.set_yscale('log')
    plotString = plotString + "_LogarithmicYAxis"

if args.plotGround == True:
    groundDepth = FindGroundSlantDepth(fileToRead, observatory="Auger")
    yMin, yMax = ax.get_ylim()
    ax.vlines(groundDepth, yMin, yMax, colors='black', linestyles='--', label='Ground')
    plotString = plotString + "_wGroundLine"

ax.legend(loc="best", fontsize=10, ncol=1)

filename = ABS_PATH_HERE + "/plots/LongDistribution_IndividualParticleTypeCurves" + plotString + ".pdf"
plt.savefig(filename, bbox_inches="tight")
print("Saved", filename)

print(f"R (corsika): {Rcorsika:.3f}")
print(f"L (corsika): {Lcorsika:.3f}")
print(f"Xmax (corsika): {xmax:.3f}")

