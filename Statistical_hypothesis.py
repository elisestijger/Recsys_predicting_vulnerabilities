
from scipy import stats
import numpy as np

def pairedt(scoresA, scoresB):
# Example performance scores for algorithms A and B

    # Perform a paired t-test
    t_statistic, p_value = stats.ttest_rel(scoresA, scoresB)

    # Output the results
    print("Paired t-test Results:")
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    # Determine the significance
    alpha = 0.05  # Set your desired level of significance
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in performance scores.")
    else:
        print("Failed to reject the null hypothesis: No significant difference in performance scores.")
    return p_value