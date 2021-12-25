import sys
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]
    data = pd.read_json(searchdata_file, orient='records', lines=True)
    
    even = data[data.uid % 2 == 0]
    odd = data[data.uid % 2 != 0]
    
    searched_atleast_once_even = list(filter(lambda x: (x >= 1), even.search_count))
    searched_atleast_once_odd = list(filter(lambda x: (x >= 1), odd.search_count))
    
    contingency = [[len(searched_atleast_once_even), len(even.uid) - len(searched_atleast_once_even)], 
                    [len(searched_atleast_once_odd), len(odd.uid) - len(searched_atleast_once_odd)] ] 
    #contingency: [[even >= 1 search, even no search], [odd >= 1 search, odd no search]]
    
    chi2_1, p_chi_1, dof_chi_1, expected_chi_1 = stats.chi2_contingency(contingency)
    man_whitney_1 = stats.mannwhitneyu(even.search_count, odd.search_count)
    
    #instructor
    even = even.drop(even[even.is_instructor == False].index)
    odd = odd.drop(odd[odd.is_instructor == False].index)
    
    searched_atleast_once_even = list(filter(lambda x: (x >= 1), even.search_count))
    searched_atleast_once_odd = list(filter(lambda x: (x >= 1), odd.search_count))
    
    contingency = [[len(searched_atleast_once_even), len(even.uid) - len(searched_atleast_once_even)], 
                    [len(searched_atleast_once_odd), len(odd.uid) - len(searched_atleast_once_odd)] ] # [[even >= 1 search, even no search], [odd >= 1 search, odd no search]]
    
    chi2_2, p_chi_2, dof_chi_2, expected_chi_2 = stats.chi2_contingency(contingency)
    man_whitney_2 = stats.mannwhitneyu(even.search_count, odd.search_count)

    print(OUTPUT_TEMPLATE.format(
        more_users_p=p_chi_1,
        more_searches_p=man_whitney_1.pvalue,
        more_instr_p=p_chi_2,
        more_instr_searches_p=man_whitney_2.pvalue,
    ))

if __name__ == '__main__':
    main()
