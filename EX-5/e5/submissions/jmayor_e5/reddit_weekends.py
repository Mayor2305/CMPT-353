import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def whatDay (d):
    day = d.date.weekday()
    
    if day == 5 or day == 6:
        return 1
    else:
        return 0
   
def centralLimitThreorem_year(data):
    iso = data.date.isocalendar()
    return iso[0]

def centralLimitThreorem_week(data):
    iso = data.date.isocalendar()
    return iso[1]

OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)

def main():
    reddit_counts = sys.argv[1]
    counts = pd.read_json(reddit_counts, lines=True)
    counts.date = pd.to_datetime(counts.date)
    counts = counts.drop(counts[counts.date.dt.year > 2013].index)
    counts = counts.drop(counts[counts.date.dt.year < 2012].index)
    counts = counts.drop(counts[counts.subreddit != "canada"].index)
    
    counts['type'] = counts.apply(whatDay, axis = 1)
    
    weekend_df = counts.drop(counts[counts.type == 0].index)
    weekday_df = counts.drop(counts[counts.type == 1].index)
    
    #initial
    isNormal_init_1 = stats.normaltest(weekend_df.comment_count).pvalue
    isNormal_init_2 = stats.normaltest(weekday_df.comment_count).pvalue
    levene_test_init = stats.levene(weekend_df.comment_count, weekday_df.comment_count).pvalue
    ttest_init = stats.ttest_ind(weekend_df.comment_count, weekday_df.comment_count).pvalue
    
    #fix1
    weekend_df.comment_count = (weekend_df.comment_count)
    weekday_df.comment_count = (weekday_df.comment_count)
    
    isNormal_fix1_1 = stats.normaltest(np.sqrt(weekend_df.comment_count)).pvalue
    isNormal_fix1_2 = stats.normaltest(np.sqrt(weekday_df.comment_count)).pvalue
    levene_test_fix1 = stats.levene(np.sqrt(weekend_df.comment_count), np.sqrt(weekday_df.comment_count)).pvalue
    
    #fix2
    weekend_df['year'] = weekend_df.apply(centralLimitThreorem_year, axis = 1)
    weekend_df['week'] = weekend_df.apply(centralLimitThreorem_week, axis = 1)
    
    weekday_df['year'] = weekday_df.apply(centralLimitThreorem_year, axis = 1)
    weekday_df['week'] = weekday_df.apply(centralLimitThreorem_week, axis = 1)
    
    mean_weekday = weekday_df.groupby(['year', 'week']).mean()
    mean_weekend = weekend_df.groupby(['year', 'week']).mean()
    
    isNormal_fix2_1 = stats.normaltest(mean_weekday.comment_count).pvalue
    isNormal_fix2_2 = stats.normaltest(mean_weekend.comment_count).pvalue
    levene_test_fix2 = stats.levene(mean_weekend.comment_count, mean_weekday.comment_count).pvalue
    
    ttest_fix2 = stats.ttest_ind(mean_weekday.comment_count, mean_weekend.comment_count).pvalue
    
    #fix3
    utest_1 = stats.mannwhitneyu(weekday_df.comment_count, weekend_df.comment_count).pvalue
    utest_2 = stats.mannwhitneyu(weekend_df.comment_count, weekday_df.comment_count).pvalue
    
    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p = ttest_init,
        initial_weekday_normality_p = isNormal_init_2,
        initial_weekend_normality_p = isNormal_init_1,
        initial_levene_p = levene_test_init,
        transformed_weekday_normality_p = isNormal_fix1_2,
        transformed_weekend_normality_p = isNormal_fix1_1,
        transformed_levene_p = levene_test_fix1,
        weekly_weekday_normality_p = isNormal_fix2_1,
        weekly_weekend_normality_p = isNormal_fix2_2,
        weekly_levene_p = levene_test_fix2,
        weekly_ttest_p = ttest_fix2,
        utest_p = utest_1,
    ))
        
if __name__ == '__main__':
    main()
