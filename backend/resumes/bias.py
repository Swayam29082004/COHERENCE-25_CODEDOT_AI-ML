import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame

def avg_combined_score(y_true, y_pred):
    return np.mean(y_pred)

def calculate_bias_metrics(top_candidates):
    # Convert list of candidate dicts to a DataFrame.
    df = pd.DataFrame(top_candidates)
    
    # Ensure the DataFrame has a sensitive feature. Here we assume 'gender'.
    if 'gender' not in df.columns:
        # For demonstration, simulate a gender column.
        df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
    
    # Create dummy y_true array of floats with same length.
    y_true = np.ones(len(df), dtype=float)
    
    # Ensure y_pred is a numeric numpy array.
    y_pred = df["combined_score"].to_numpy(dtype=float)
    
    # Ensure sensitive features is a numpy array.
    sensitive_features = df["gender"].to_numpy()
    
    # Compute the average combined score for each gender.
    metric_frame = MetricFrame(
        metrics={"average_combined_score": avg_combined_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Output fairness metrics by group.
    print("Fairness metrics by gender:")
    print(metric_frame.by_group)
    
    # Calculate the percentage of each group among the top candidates.
    group_percentages = df["gender"].value_counts(normalize=True) * 100
    print("Bias percentages among top candidates:")
    print(group_percentages)
    
    return metric_frame, group_percentages

# Example top 5 candidate data:
top_candidates_example = [
    {"name": "Candidate1", "gender": "Male", "combined_score": 85.0, "job_fit_score": 88, "similarity_with_new": 80},
    {"name": "Candidate2", "gender": "Female", "combined_score": 82.0, "job_fit_score": 83, "similarity_with_new": 80},
    {"name": "Candidate3", "gender": "Male", "combined_score": 80.0, "job_fit_score": 80, "similarity_with_new": 80},
    {"name": "Candidate4", "gender": "Female", "combined_score": 78.0, "job_fit_score": 78, "similarity_with_new": 80},
    {"name": "Candidate5", "gender": "Male", "combined_score": 75.0, "job_fit_score": 75, "similarity_with_new": 80},
]

calculate_bias_metrics(top_candidates_example)
