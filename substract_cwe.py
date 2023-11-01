import pandas as pd     #need version 1.4.0 ! 

def extract_cwe(cve_all):
    cwe_all = cve_all
    values = []
    for data in cwe_all['cve.problemtype.problemtype_data']:
        try:
            descriptions = data[0]['description']
            cwe_list = [desc['value'] for desc in descriptions]
            values.append(cwe_list)
        except (IndexError, KeyError):
            values.append([]) 
    cwe_all['cwe'] = values

    # remove unknown products and reset index 
    cwe_all = cwe_all[cwe_all['product']!='unknown'] 
    cwe_all.reset_index(drop=True, inplace=True)

    # get all cwe's even the doubles 
    cwe_all_exploded = cwe_all.explode('cwe', ignore_index=True)
    cwe_all_exploded_new=cwe_all_exploded.dropna(subset=['cwe'])

    return cwe_all_exploded_new