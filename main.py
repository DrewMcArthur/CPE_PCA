import itertools
import os
from typing import List, Dict, Tuple, Optional

from pandas import DataFrame, Series, read_csv, to_numeric, concat, set_option
from numpy import nan, log1p, sqrt

from sklearn.decomposition import PCA  
from scipy.stats import pearsonr, skew, kurtosis

import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor  


from constants import (
    input_dir,
    input_file,
    output,
    DVs,
    IVs,
    IVsPerDvForMultivariates,
    InteractionVarsPerDv,
)
from columns import Columns as C

VERT_SEP = "============================================================================================"


def main():
    df = import_data()
    print(VERT_SEP)
    do_correlations(df)
    print(VERT_SEP)
    do_regressions(df)
    print(VERT_SEP)


def import_data():
    """ imports the datafile to use.  see `input_example.csv` for an example format """
    file_path = f"{input_dir}/{input_file}"
    raw_data = read_csv(file_path, header=None, na_values=["#VALUE!", "#N/A","N/A", "..", ""])

    # Combine the first two rows as a multi-level header
    combined_headers = raw_data.iloc[0].fillna("") + " " + raw_data.iloc[1].fillna("")
    combined_headers = combined_headers.str.strip()

    df = raw_data.iloc[2:]
    df.columns = [C.parse(c) for c in combined_headers]
    return clean(df)


def clean(df: DataFrame) -> DataFrame:
    """ drop empty columns, & coerce into numeric format"""
    df = df.dropna(axis=1, how="all")
    df = df.replace(",", "", regex=True)
    for c in DVs + IVs:
        df[c] = to_numeric(df[c], errors="coerce")
    return df.replace("..", nan)


def calculate_skew_kurt(df, threshold=0.75):
    """Identify columns with high skewness or kurtosis."""
    results = {}
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            col_skew = skew(df[col].dropna())
            col_kurt = kurtosis(df[col].dropna())
            if abs(col_skew) > threshold or abs(col_kurt) > threshold:
                results[col] = {'skew': col_skew, 'kurtosis': col_kurt}
    return results

def transform_columns(df, transform_type='log1p'):
    """Apply transformations to reduce skewness and kurtosis."""
    for col, metrics in calculate_skew_kurt(df).items():
        print(f"Transforming {col}: Skew = {metrics['skew']}, Kurtosis = {metrics['kurtosis']}")
        if transform_type == 'log1p':
            df[col] = log1p(df[col].clip(lower=0))
        elif transform_type == 'sqrt':
            df[col] = sqrt(df[col].clip(lower=0))
        elif transform_type == 'boxcox':
            from scipy.stats import boxcox
            df[col], _ = boxcox(df[col].clip(lower=1))
    return df

def do_correlations(df: DataFrame):
    """ runs correlations on each pair of variables, and prints out the results """
    pairs = list(itertools.combinations(DVs+IVs, 2))
    correlations = DataFrame([get_correlation(df, a, b) for (a, b) in pairs])

    if not os.path.exists(output):
        os.makedirs(output)
    correlations.to_csv(f"{output}/correlations.csv")

    correlations["significant"] = correlations["p"] < 0.01
    correlations["is_coop_var"] = correlations["a"].isin(DVs) | correlations["b"].isin(DVs)

    sig_cors = correlations[correlations["significant"]]
    sig_coop_cors = sig_cors[sig_cors["is_coop_var"]]
    print(f"{len(sig_cors)} significant correlations, {len(sig_coop_cors)} of which are coop vars")
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    print(sig_cors.sort_values(by=["is_coop_var", "a", "p"], ascending=True, ))


def get_correlation(df: DataFrame, a: str, b: str):
    """given a dataframe, a dependent variable, and an independent variable,
    print out the correlation between the two variables
    """
    df = df[[a, b]]
    df = df.dropna(axis=0, how="any")
    n = len(df)
    corr, p_val = pearsonr(df[a], df[b])
    return {"a": a, "b": b, "corr": corr, "p": p_val, "n": n}


def do_regressions(df: DataFrame):
    """ runs a regression for each DV """
    df = fix_skews(df)
    models = {dv: run_regression(df, dv) for dv in DVs}
    export_models(models)

def fix_skews(df: DataFrame) -> DataFrame:
    """ log transform columns with high skewness """
    skewedness = check_skews(df)
    for c, v in skewedness.items():
        if v > 0:
            df[c] = log1p(df[c])
    return df


def export_models(
    models: Dict[C, Optional[sm.OLS]], output_path: str = f"{output}/regression_results.csv"
):
    """
    Exports regression results (coefficients, p-values, R-squared, etc.) to a CSV file.

    Parameters:
        models (Dict[str, sm.OLS]): Dictionary of dependent variable names to fitted regression models.
        output_path (str): Path to save the CSV file.
    """
    results = []

    for dv, model in models.items():
        if model is None:
            continue
        summary = model.summary2().tables[1]
        for var in summary.index:
            coef = summary.loc[var, "Coef."]
            pval = summary.loc[var, "P>|t|"]
            conf_lower = summary.loc[var, "[0.025"]
            conf_upper = summary.loc[var, "0.975]"]
            results.append(
                {
                    "Dependent Variable": dv,
                    "Independent Variable": var,
                    "Coefficient": coef,
                    "P-Value": pval,
                    "Confidence Interval (Lower)": conf_lower,
                    "Confidence Interval (Upper)": conf_upper,
                    "R-Squared": model.rsquared,
                    "Adjusted R-Squared": model.rsquared_adj,
                }
            )

    results_df = DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Regression results exported to {output_path}")


def check_skews(df: DataFrame) -> Dict[C, float]:
    """returns -1, 0, or 1 for each column representing negative (left) or positive (right) skew"""
    return {
        C.parse(c): is_skewed(df[c]) for c in df.columns if df[c].dtype == "float64"
    }


def is_skewed(col: Series) -> int:
    """ returns 0 if not skewed, or the skew value if abs. value is greater than 1 """
    res = skew(col.dropna())
    if -1 < res < 1:
        return 0
    return int(res)


def run_regression(df: DataFrame, dv: C) -> Optional[sm.OLS]:
    """ runs a regression for the given DV """
    ivs = IVsPerDvForMultivariates.get(dv, [])
    if len(ivs) == 0:
        print(f"No IVs for dv ({dv})")
        return None
    df = fix_skews(df)
    df = fix_outliers(df, dv, ivs) 

    cols = [str(iv) for iv in ivs]
    df, cols = add_interaction_vars(df, dv, cols)
    df = df[[str(dv)] + cols].dropna(axis=0, how="any")

    if len(df) < len(cols) + 3 or len(df) < 20 or len(cols) == 0:
        print(f"Not enough data to run regression for dv ({dv}) cols: ", cols)
        return None

    if C.Cooperative_Specific_Legislation in cols:
        # square ordinal variable
        df[C.Cooperative_Specific_Legislation] = df[C.Cooperative_Specific_Legislation] ** 2
    
    X = df[cols]
    y = df[dv]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model

def fix_outliers(df: DataFrame, dv: str, ivs: List[C]) -> DataFrame:
    """ calculates VIF and adjusts variables accordingly, subtracting the mean """
    assert len(ivs) > 0
    vifs = calculate_vif(df, ivs)
    if vifs is not None:
        for _, row in vifs.iterrows():
            iv = row["Variable"]
            vif = row["VIF"]
            if iv in df.columns and vif > 5:
                print(f"High VIF for {iv}: {vif}, fixing.")
                df[iv] = df[iv] - df[iv].mean()
    return df

def add_interaction_vars(df: DataFrame, dv: C, cols: List[str]) -> Tuple[DataFrame, List[str]]:
    """ adds interaction variables to dataframe and column list """
    interaction_vars = gen_interaction_vars(df, InteractionVarsPerDv.get(dv))
    if interaction_vars is not None:
        df = concat([df, DataFrame(interaction_vars)], axis=1)
        cols += list(interaction_vars.keys())
    return df, cols

def calculate_vif(df: DataFrame, ivs: List[C]) -> Optional[DataFrame]:
    """ returns the VIF for each variable in the given iv list """
    df = df[ivs].dropna()
    if len(df) < len(ivs) + 3:
        print("Not enough data to calculate VIF for ivs: ", ivs)
        return None
    X = sm.add_constant(df[ivs])
    vif = DataFrame(
        {
            "Variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    return vif


def gen_interaction_vars(
    df: DataFrame, pairs: Optional[List[Tuple[C, C]]]
) -> Optional[Dict[str, Series]]:
    """ given a dataframe and a list of pairs of independent variables,
        return a dictionary of interaction variables values by name """
    if pairs is None:
        return None
    interaction_vars = {}
    for a,b in pairs:
        valid_rows = df[[a, b]].dropna()

        # normalize inputs for interaction
        valid_rows[a] = valid_rows[a] - valid_rows[a].mean()
        valid_rows[b] = valid_rows[b] - valid_rows[b].mean()

        interaction_vars[f"{a}*{b}"] = valid_rows[a] * valid_rows[b]
    return interaction_vars


def do_pca(df: DataFrame, dv: str, ivs: List[str]):
    """given a dataframe, a dependent variable, and a list of independent variables,
    perform a pca and print out the loadings and variance explained
    """
    print(df)
    data = df[[dv] + ivs]
    data_array = data.values
    pca = PCA(n_components=len(ivs))
    pca_data = pca.fit_transform(data_array)
    pca_df = DataFrame(pca_data, columns=ivs)
    pca_df[dv] = df[dv]
    # print the explained variance ratio for each component
    print("Explained Variance Ratio:")
    for i, iv in enumerate(ivs):
        print(f"{iv}: {pca.explained_variance_ratio_[i]}")

    return pca_df, pca


if __name__ == "__main__":
    main()
