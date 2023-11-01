from urllib.parse import urlencode 
from urllib.request import Request, urlopen
import pandas as pd

url			= 'https://vuldb.com/?api'									#url endpoint
post_fields	= { 'apikey': '[7fea568c5475142acd1411bd3befe8cd]', 'recent': '100000' }	#request

request = Request(url, urlencode(post_fields).encode())
json = urlopen(request).read().decode()
# print(json)


    # Extract CVE information
    cve_list = []
    for entry in json['result']:
        cve_id = entry['entry']['id']
        cve_title = entry['entry']['title']
        cve_list.append({'CVE ID': cve_id, 'Title': cve_title})

    # Create a DataFrame from the CVE information
    df = pd.DataFrame(cve_list)

    # Save the DataFrame as a CSV file
    df.to_csv('cves_2022CHECK.csv', index=False)

    print('CVEs in 2022 saved successfully as cves_2022CHECK.csv.')
else:
    print('Error: Failed to retrieve CVEs from the API.')

