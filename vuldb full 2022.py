import requests
import json
import pandas as pd

# Define your personal API key here
personalApiKey = 'ea3eba19503f389e553529f5f78eac92'


# Set HTTP Header
userAgent = 'VulDB API Advanced Python Demo Agent'
headers = {'User-Agent': userAgent, 'X-VulDB-ApiKey': personalApiKey}

# VulDB API endpoint
url = 'https://vuldb.com/?api'

# Define the API call to retrieve the last 1000 results
postData = {'recent': 1000, 'details': 1}
response = requests.post(url, headers=headers, data=postData)

if response.status_code == 200:
    responseJson = json.loads(response.content)
    results = responseJson.get('result')

    if results:
        # Save the results as a JSON file
        with open('vuldb_results.json', 'w') as f:
            json.dump(results, f)

        print('Results saved successfully as vuldb_results.json.')
    else:
        print('No results found.')
else:
    print('Error: Failed to retrieve results from the VulDB API.')
