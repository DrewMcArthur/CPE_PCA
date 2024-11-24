# CPE Stats Calculations

The code here was used to calculate correlations and regressions
for my final paper in PSCI 7312, Comparative Political Economy.  

I explored relations between various country level statistics 
(% GDP in certain sectors, Income Inequality, indices for social capital and public spending, etc.)
and metrics of cooperative prevalence (number of cooperatives, employees, members, all per million)

The code here, given input data in csv format, will calculate correlations for all variables given
and regressions for specific variable confiurations defined in `constants.py`.

## Files

- `main.py` contains the bulk of the script's logic. 
- `columns.py` defines an enum for the columns expected from the input csv.
- `constants.py` contains various constants and configuration for the script.

## Running the code

1. First, populate the input datafile.  
2. store it in `{CONSTANTS.input_dir}/{CONSTANTS.input_file}`.
3. Adjust any of the configuration or logic as you see fit.
4. `pipenv install`
5. `pipenv run python main.py`