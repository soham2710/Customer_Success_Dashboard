import pandas as pd
import numpy as np
from autoviz.AutoViz_Class import AutoViz_Class
from scipy import stats
from sklearn.datasets import load_iris
from pycaret.classification import setup, compare_models

# Load and preprocess the Iris dataset
def load_and_preprocess_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    # Create control and test groups based on 'target' column
    df['Group'] = np.where(df['target'] < 1, 'Control', 'Test')
    return df

# Auto visualization
def auto_visualize(df):
    AV = AutoViz_Class()
    AV.AutoViz(df)

# Statistical tests
def run_statistical_tests(df):
    results = {}
    control = df[df['Group'] == 'Control']['sepal length (cm)']
    test = df[df['Group'] == 'Test']['sepal length (cm)']
    
    # t-test
    t_stat, t_p = stats.ttest_ind(control, test)
    results['t-test'] = (t_stat, t_p)
    
    # z-score
    z_score = (np.mean(test) - np.mean(control)) / np.std(df['sepal length (cm)'])
    results['z-score'] = z_score
    
    # F-test
    f_stat, f_p = stats.f_oneway(control, test)
    results['f-test'] = (f_stat, f_p)
    
    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(control, test)
    results['mann-whitney'] = (u_stat, u_p)
    
    # Chi-square test
    chi2_stat, chi2_p = stats.chisquare(df['target'])
    results['chi-square'] = (chi2_stat, chi2_p)
    
    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(control, test)
    results['pearson'] = (pearson_corr, pearson_p)
    
    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(control, test)
    results['spearman'] = (spearman_corr, spearman_p)
    
    # ANOVA
    anova_stat, anova_p = stats.f_oneway(control, test)
    results['anova'] = (anova_stat, anova_p)
    
    # Monte Carlo simulation (example)
    monte_carlo = np.mean([np.mean(np.random.choice(df['sepal length (cm)'], size=len(control))) for _ in range(1000)])
    results['monte-carlo'] = monte_carlo

    return results

# PyCaret model comparison
def run_pycaret_analysis(df):
    exp = setup(df, target='target', silent=True, html=False)
    best_model = compare_models()
    return best_model

if __name__ == "__main__":
    df = load_and_preprocess_data()
    auto_visualize(df)
    stats_results = run_statistical_tests(df)
    best_model = run_pycaret_analysis(df)
    print(stats_results)
    print(best_model)
