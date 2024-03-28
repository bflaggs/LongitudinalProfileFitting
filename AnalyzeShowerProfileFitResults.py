#!/usr/bin/env python3

######################################
# Description of file and file usage #
######################################

# File to analyze the longitudinal profile fits
# Used to determine which fit type has the best results and total number of fits to cut from the analysis

# Run by executing the command...
# ./AnalyzeShowerProfileFitResults.py PATH_TO_ASCII_FILES --kwargs

# PATH_TO_ASCII_FILES for diffrent experiments:
# IceCube -> /home/acoleman/gen2-surface/sim/mass/*.txt
# Auger   -> /pbs/home/b/bflaggs/SimulationWork/ParsedData/SIB23c/*/FILES_TO_READ

# Possible Updates to Make:
# 1. Add a way to make histograms of the uncertainties in the longitudinal fit parameters
# 2. Add a way to make Xmax comparison scatter plots which plot fit Xmax vs. CORSIKA Xmax

######################
# End of description #
######################

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import rc, rcParams

rc("font", size=18.0)
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "dejavuserif"

import numpy as np
import os
from os import listdir
from os.path import isfile, join

ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))

from python_tools.PlottingTools import qualitative_colors
from python_tools.MultivariateMass_FinalizedAnalysis_FOMPaper import MultivariateMassAnalysis

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, nargs="+", default=[], help="List of CORSIKA simulation ASCII files")
parser.add_argument("--observatory", type=str, nargs="?", required=True, default="IceCube", help="Name of observatory (either IceCube or Auger)")
parser.add_argument("--zenithRange", type=float, nargs=2, default=[0.0, 71.6], help="Zenith range of data to plot")
parser.add_argument("--energyRange", type=float, nargs=2, default=[16.0, 18.5], help="lg(E) energy range of data to plot")
parser.add_argument("--compareFitTypes", action="store_true", help="If set, will make plots to compare number of poor fits to GH shifted and parameterized functions")
parser.add_argument("--printNumberEventsToCut", action="store_true", help="If set, will print the number of events to cut from analysis for specific fit")
parser.add_argument("--makeLongFitHistograms", action="store_true", help="If set, will make histograms of longitudinal fit parameters")
parser.add_argument("--useGHShiftedFits", action="store_true", help="If set, will use the GH shifted fits in analysis (making histograms or printing poor fit numbers)")
parser.add_argument("--applyDataCuts", action="store_true", help="If set, will cut out fits with unconstrained/poor fit parameters")
args = parser.parse_args()

if args.observatory != "IceCube" and args.observatory != "Auger":
    raise ValueError("Settings not currently allowed for observatories other than IceCube or Auger.")

if args.zenithRange[0] >= args.zenithRange[1]:
    raise ValueError("Can not have starting zenith angle be greater than or equal to final zenith angle.")

if args.energyRange[0] >= args.energyRange[1]:
    raise ValueError("Can not have starting energy bin be greater than or equal to final energy bin.")

if args.useGHShiftedFits == True and args.printNumberEventsToCut + args.makeLongFitHistograms == 0:
    print("Warning: Using the GH Shifted Fits does nothing unless printing to terminal the number of poor fits or making histograms.\n")

minDeg = args.zenithRange[0]
maxDeg = args.zenithRange[1]

minLgE = args.energyRange[0]
maxLgE = args.energyRange[1]

observatory = args.observatory
flagGHShiftedFits = args.useGHShiftedFits
flagDataCut = args.applyDataCuts

if flagDataCut == True:
    fileDataCut = "_DataCutsApplied"
else:
    fileDataCut = ""

flagAllPrims = True # Note: Can not make Fisher projections when this flag is set to "True"
flagProtHe = False
flagHeO = False

if flagAllPrims == True and flagProtHe == False and flagHeO == False:
    filePrimNames = "_allPrimaries"
elif flagAllPrims == False and flagProtHe == True and flagHeO == False:
    filePrimNames = "_ProtonHelium"
elif flagAllPrims == False and flagProtHe == False and flagHeO == True:
    filePrimNames = "_HeliumOxygen"
else:
    filePrimNames = "_ProtonIron"


analysis = MultivariateMassAnalysis(minDeg=minDeg, maxDeg=maxDeg, muonScaling=0.0, highEmuonScaling=0.0, minLgE=minLgE, maxLgE=maxLgE,
                                    includeXmax=True, includeMuObsLev=False, includeMuhighE=False, includeMu800m=False,
                                    includeEMratio=False, includeEMxmax=False, includeEMObslev=False,
                                    includeRval=True, includeLval=True, useGHFits=flagGHShiftedFits, useCorsikaXmax=False,
                                    first10rings=False, last10rings=False, allfourPrimaries=flagAllPrims,
                                    protonAndHelium=flagProtHe, heliumAndOxygen=flagHeO, applyScaling=False, applyDataCuts=flagDataCut,
                                    observatory=observatory, useLargerSmearValues=False, singleObservable=False, smearVal=0.0)

for file in args.input:
    analysis.ReadSingleFile(file)

if args.compareFitTypes == True:
    # Running these class functions w/ the applyDataCuts keyword does not change the output plots...
    analysis.PlotBadFitsFractions(ABS_PATH_HERE, includeBadFitValues=True)
    analysis.PlotTypesBadFits(ABS_PATH_HERE, fittype="GHShifted")
    analysis.PlotTypesBadFits(ABS_PATH_HERE, fittype="Andringa")

if args.printNumberEventsToCut == True:
    if args.useGHShiftedFits == True:
        print("Printing statistics of poor events for the GH shifted fits...\n")
    else:
        print("Printing statistics of poor events for the GH parameterized fits...\n")
    analysis.GetNumberOfCutEvents()
    analysis.GetNumberOfEventsWhereCORSIKAFitFails()


if args.makeLongFitHistograms == True:
    analysis.GetValues()

    if args.useGHShiftedFits == True:
        filename = ABS_PATH_HERE + "/plots/histograms/FitParameters/GHShiftedFit/" + observatory + "_LongitudinalFit" + filePrimNames + fileDataCut + \
                 f"_lgE{minLgE:.1f}_{maxLgE:.1f}_zen{minDeg:.0f}_{maxDeg:.0f}"
    else:
        filename = ABS_PATH_HERE + "/plots/histograms/FitParameters/GHParameterizedFit/" + observatory + "_LongitudinalFit" + filePrimNames + fileDataCut + \
                 f"_lgE{minLgE:.1f}_{maxLgE:.1f}_zen{minDeg:.0f}_{maxDeg:.0f}"

    # Can fix plot range w/ fixedBins and include mean+median+mode in legend w/ calcMeanMedianMode 
    analysis.MakeHistograms(filename, fixedBins=False, calcMeanMedianMode=False)

