# model.py

import pandas as pd
import numpy as np
from autoviz.AutoViz_Class import AutoViz_Class
from scipy import stats
import pycaret.classification as pycaret_clf

# Function to load and explore dataset
def load_dataset(file):
    return pd.read_csv(file)

# Function for statistical tests
def perform_statistical_tests(df):
    results = {}
    
    # Example: T-test
    t_stat, t_p = stats.ttest_ind(df['group1'], df['group2'])
    results['T-test'] = {'Statistic': t_stat, 'p-value': t_p}
    
    # Example: Chi-square test
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(pd.crosstab(df['category1'], df['category2']))
    results['Chi-square test'] = {'Statistic': chi2_stat, 'p-value': chi2_p}
    
    # F-test
    f_stat, f_p = stats.f_oneway(df['group1'], df['group2'])
    results['F-test'] = {'Statistic': f_stat, 'p-value': f_p}
    
    # Z-score
    df['z_score'] = stats.zscore(df['value_column'])  # Replace 'value_column' with your column
    results['Z-score'] = df['z_score'].tolist()
    
    # ANOVA
    anova_stat, anova_p = stats.f_oneway(df['group1'], df['group2'], df['group3'])
    results['ANOVA'] = {'Statistic': anova_stat, 'p-value': anova_p}
    
    # Mann-Whitney U Test
    u_stat, u_p = stats.mannwhitneyu(df['group1'], df['group2'])
    results['Mann-Whitney U Test'] = {'Statistic': u_stat, 'p-value': u_p}
    
    # Kruskal-Wallis H Test
    h_stat, h_p = stats.kruskal(df['group1'], df['group2'], df['group3'])
    results['Kruskal-Wallis H Test'] = {'Statistic': h_stat, 'p-value': h_p}
    
    # Wilcoxon Signed-Rank Test
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(df['group1'], df['group2'])
    results['Wilcoxon Signed-Rank Test'] = {'Statistic': wilcoxon_stat, 'p-value': wilcoxon_p}
    
    # Friedman Test
    friedman_stat, friedman_p = stats.friedmanchisquare(df['group1'], df['group2'], df['group3'])
    results['Friedman Test'] = {'Statistic': friedman_stat, 'p-value': friedman_p}
    
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(df['value_column'])
    results['Shapiro-Wilk Test'] = {'Statistic': shapiro_stat, 'p-value': shapiro_p}

    return results

# Monte Carlo Simulation
def monte_carlo_simulation(df, n=1000):
    simulations = []
    for _ in range(n):
        sample = df.sample(frac=1, replace=True)
        simulations.append(sample.mean())
    return simulations

# PyCaret Model
def run_pycaret(df, target):
    pycaret_clf.setup(data=df, target=target, silent=True)
    best_model = pycaret_clf.compare_models()
    return best_model, pycaret_clf.pull()
