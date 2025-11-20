# Code associated with McKinnon and Simpson, "Observed and modeled trends in downward surface shortwave radiation over land: drivers and discrepancies", under review in GRL.

Before running the code, it is necessary to obtain the relevant datasets, all at monthly-mean resolution: 
- ERA5 (downward surface shortwave, clearsky downward shortwave, cloud fractions)
- MERRA2 (downward surface shortwave, AOD)
- JRA-3Q (downward surface shortwave)
- CERES-EBAF ED4.2 (downward surface shortwave, TOA SW)
- CM SAF CLARA-A3 (downward surface shortwave)
- NASA GEWEX-SRB (downward surface shortwave)
- ISCCP (cloud fractions)
- GEBA surface radiation
- CESM2-LE (downward surface shortwave)
- CESM2-AMIP (downward surface shortwave)

DOIs can be found in the "Open Research" section of the paper.

The CMIP6 data is accessed and processed in cmip-pangeo-process (either in notebooks/ or scripts/ folder). This should be run for for rsds and clt.

The notebook and script paper-figs then performs all the analysis and makes all the figures.

The environment.yml can be used to reproduce the coding environment. 

Both scripts also rely on https://github.com/karenamckinnon/helpful_utilities

Please email Prof. Karen McKinnon (kmckinnon@ucla.edu) with questions or comments.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
