import db
import requests
import json

def get_labels():
    # Get the succesful domains for building a query to the labelling API
    successful_domains = db.get_succesful_domain_urls()

    # Turn successful domain URLs into a GET request to the labeling API

    url = "https://api-url.cyren.com/api/v1/free/urls-list"

    payload = json.dumps({
        "urls": successful_domains
    })

    headers = {
    'Authorization': 'Bearer your.jwt.token',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
