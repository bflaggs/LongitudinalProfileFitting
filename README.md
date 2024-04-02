# LongitudinalProfileFitting

Repository holding all the code I used for fitting longitudinal profiles from CORSIKA simulations. A lot of this was used for fitting the profiles in the [mass separation analysis](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.042002) but with updates to make it more modular and allowing it to be separate from the mass separation/sensitivity analysis.

The description and usage of most files should be given in the beginning of almost all files.

Things to update:
1. Add a submission script file that can read in all .long files, do the fits, and save .txt files.
2. Clean up plotting functions, maybe condense them to one file
3. Add my scaling script

Becasue this code comes from the mass separation analysis, there is some code structure based on code from Alan Coleman. So thanks Alan :)
