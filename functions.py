import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest



def two_sample_z_test(dataset_1_part, dataset_1_total, dataset_2_part, dataset_2_total):
    z_score, p_value = proportions_ztest(
        [dataset_1_part, dataset_2_part], [dataset_1_total, dataset_2_total ], alternative='two-sided')
    significance = ""
    if p_value < 0.05:
        significance = "Yes."
    else:
        significance = "No."

    print("Z-score:", z_score)
    print("P-value:", p_value)
    print("Is it statistically significant? ",significance)


def chi2(data_attribute, data_preference):
    contingency_table = pd.crosstab(data_attribute, data_preference)
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square Test Results for {data_attribute.name} vs {data_preference.name}")
    print("Chi-square statistic:", chi2)
    print("p-value:", p_value)

    