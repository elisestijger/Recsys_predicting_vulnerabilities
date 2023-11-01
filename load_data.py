import os
import sys
import numpy as np
import pandas as pd     #need version 1.4.0 ! 
from pandas import json_normalize 
sys.path.append(os.getcwd())
import joblib


def get_data(year1,year2):
    cve_all = load_data(year1,year2)
    cve_all['description'] = cve_all.apply(lambda row: row['cve.description.description_data'][0]['value'], axis=1)
    cve_all[['vendor', 'product']]=cve_all.apply(lambda row: get_vendor(row), axis=1)
    return cve_all

def save_data():
    cve_all = get_data()
    joblib.dump(cve_all, 'cve_all.pkl')


#faster way to download data
def grab_data():
    loaded_data = joblib.load('cve_all.pkl')
    return loaded_data

def load_data(year1,year2):
    # Load CVEs from the following years
    CVE_YEARS = []
    for num in range(year1, year2 + 1):  
        CVE_YEARS.append(str(num)) 

    print(CVE_YEARS)
    # CVE_YEARS = ['2005', '2006','2007', '2008', '2009', '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2022','2023']
    cve_all_raw = pd.DataFrame()
    for cveyear in CVE_YEARS:
        print(cveyear)
        CVE_FEED = 'nvdcve-1.1-' + cveyear + '.json'
        # Without the zip files you can retrieve data from api:
        # CVE_URL = 'https://nvd.nist.gov/feeds/json/cve/1.1/' + CVE_FEED + '.zip'
        # !wget -qq $CVE_URL
        # !unzip $CVE_FEED
        cvejson = pd.read_json(CVE_FEED)
        cve_all_raw = cve_all_raw.append(pd.json_normalize(cvejson['CVE_Items']))
        print(cve_all_raw['publishedDate'].iloc[-1])
        cve_all = cve_all_raw
    return cve_all

def get_vendor(row):
    if len(row['configurations.nodes']) > 0:
        if len(row['configurations.nodes'][0].get('cpe_match')) > 0:
            return pd.Series(row['configurations.nodes'][0].get('cpe_match')[0]['cpe23Uri'].split(":")[3:5])
        else:
            return pd.Series(row['configurations.nodes'][0].get('children')[0].get('cpe_match')[0]['cpe23Uri'].split(":")[3:5])
    else:
        return pd.Series(['unknown', 'unknown'])

