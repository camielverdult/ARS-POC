import db
import requests

def get_labels():
    # Get the succesful domains for building a query to the labelling API
    successful_domains = db.get_succesful_domain_urls()

    # Turn succesful domain URLs into a GET request to the labelling API
    
