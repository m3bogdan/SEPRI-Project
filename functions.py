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

def chi2_A(results_data, survey_data):
    # Rename the unnamed column to "where_voting" as it refers to polling stations and e-votes
    results_data = results_data.rename(columns={"Unnamed: 0": "where_voting"})
    
    # Transform 'where_voting' to have consistent labels
    transformed_results_data = results_data.copy()
    transformed_results_data["where_voting"] = transformed_results_data["where_voting"].apply(
        lambda x: "Polling station" if "Polling station" in x else "E-votes"
    )
    
    # Aggregate the results data
    agg_functions = {'Red': 'sum', 'Green': 'sum', 'Invalid ballots': 'sum', 'Total': 'sum'}
    new_results_data = transformed_results_data.groupby("where_voting").aggregate(agg_functions).reset_index()
    
    # Prepare contingency table from survey data
    contingency_table = pd.crosstab(survey_data['evote'], survey_data['party'])
    contingency_table.columns.name = None
    contingency_table = contingency_table.reset_index()
    contingency_table = contingency_table.rename(columns={"evote": "where_voting", "Invalid vote": "Invalid ballots"})
    
    # Map 'where_voting' values to consistent labels
    contingency_table["where_voting"] = contingency_table["where_voting"].map({1: "E-votes_s", 0: "Polling station_s"})
    
    # Extract survey data
    evote_survey = contingency_table[contingency_table["where_voting"] == "E-votes_s"].iloc[0]
    polling_survey = contingency_table[contingency_table["where_voting"] == "Polling station_s"].iloc[0]
    
    # Adjust new_results_data labels
    new_results_data["where_voting"] = new_results_data["where_voting"].map(
        {"E-votes": "E-votes_r", "Polling station": "Polling station_r"}
    )
    new_results_data = new_results_data[["where_voting", "Red", "Green", "Invalid ballots"]]
    
    # Extract results data
    evote_results = new_results_data[new_results_data["where_voting"] == "E-votes_r"].iloc[0]
    polling_results = new_results_data[new_results_data["where_voting"] == "Polling station_r"].iloc[0]
    
    # Create dataframes for chi-square test
    df_evote = pd.DataFrame([evote_results, evote_survey]).set_index('where_voting')
    df_evote = df_evote[['Red', 'Green', 'Invalid ballots']].apply(pd.to_numeric)
    
    df_polling = pd.DataFrame([polling_results, polling_survey]).set_index('where_voting')
    df_polling = df_polling[['Red', 'Green', 'Invalid ballots']].apply(pd.to_numeric)
    
    # Perform chi-square test for polling station
    print("Polling station")
    from scipy.stats import chi2_contingency
    chi2_polling, p_value_polling, dof_polling, expected_polling = chi2_contingency(df_polling)
    print("Chi-square Test Results for Polling Station")
    print("Chi-square statistic:", chi2_polling)
    print("p-value:", p_value_polling)
    # Perform chi-square test for e-votes
    print("Evotes")
    chi2_evote, p_value_evote, dof_evote, expected_evote = chi2_contingency(df_evote)
    print("Chi-square Test Results for E-votes")
    print("Chi-square statistic:", chi2_evote)
    print("p-value:", p_value_evote)

def k_anonymity_violations(data, quasi_identifiers, k_levels=[2, 3, 5]):
    # Group by quasi-identifiers to get the counts for each unique combination
    grouped = data.groupby(quasi_identifiers).size().reset_index(name='count')
    
    # Dictionary to store the number of rows violating each k-level
    violations = {}
    total_rows = len(data)

    for k in k_levels:
        # Count rows where the group size is less than k
        violating_rows = grouped[grouped['count'] < k]['count'].sum()
        # Store as a percentage for comparison with R's output
        violations[k] = violating_rows, (violating_rows / total_rows) * 100

    return violations

def identify_k_anonymity_violations(data, quasi_identifiers, k=2):
    # Group by quasi-identifiers to count each unique combination
    grouped = data.groupby(quasi_identifiers).size().reset_index(name='count')
    
    # Filter for combinations where the count is less than k (i.e., violate k-anonymity)
    violating_combinations = grouped[grouped['count'] < k]
    
    # Merge to find the original rows that match these violating combinations
    violations = data.merge(violating_combinations.drop(columns='count'), on=quasi_identifiers, how='inner')
    
    return violations