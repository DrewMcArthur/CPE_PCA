from typing import List, Dict, Tuple
from columns import Columns as C

input_dir = "input"
input_file = "CPE Paper Data - IV Data.csv"
output = "output-all-correlations"

DVs = [
    C.COOPS_per_capita,
    C.EMPLOYEES_per_capita,
    C.WORKER_MEMBERS_per_capita,
    C.PRODUCER_MEMBERS_per_capita,
]

IVs = [
    C.Cooperative_Specific_Legislation,
    C.Union_Density,
    C.collective_bargaining_coverage_rate,
    C.natl_compliance_with_labor_rights,
    C.PCT_of_GDP_in_Agriculture,
    C.industry,
    C.manufacturing,
    C.services,
    C.Social_Capital,
    C.Income_Inequality,
    C.Education,
    C.Economic_Liberalism_Index,
    C.OECD_Social_Spending,
]

""" When doing multivariate equations (do_regressions), this defines the set of independent variables for each dependent variable """
IVsPerDvForMultivariates: Dict[C, List[C]] = {
    C.COOPS_per_capita: [
        C.collective_bargaining_coverage_rate,
        C.PCT_of_GDP_in_Agriculture,
        C.Education,
    ],
    C.EMPLOYEES_per_capita: [
        C.Cooperative_Specific_Legislation,
        C.OECD_Social_Spending,
    ],

    C.PRODUCER_MEMBERS_per_capita: [
        C.Income_Inequality,
    ],
}

InteractionVarsPerDv: Dict[C, List[Tuple[C, C]]] = {
    C.PRODUCER_MEMBERS_per_capita: [
        (C.manufacturing, C.Social_Capital),  # Manufacturing depends on social cohesion
    ],
}
