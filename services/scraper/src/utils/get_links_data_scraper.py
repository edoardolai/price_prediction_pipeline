from requests import Session
from typing import List, Optional
def get_links_from_page(url: str, session: Session):
        """
            Fetches property links from a search results page.

            Args:
                url (str): The URL of the search results page.
                session (Session): A `requests.Session` object for managing HTTP connections and cookies.

            Returns:
                Optional[List[str]]: A list of property links formatted as URLs, or `None` if the request fails.

            Raises:
                ValueError: If the response structure is missing the expected keys (e.g., 'results', 'property', 'id').
        """
        response = session.get(url)
        if response.status_code == 200:
            data = response.json()
            base_url = "https://www.immoweb.be/en/classified/{}/for-sale/{}/{}/{}/\n"
            links = [{"link": base_url.format(str(result['property']['subtype']) ,str(result['property']['location']['district']),str(result['property']['location']['postalCode']),str(result["id"])), 
                      "lat": result["property"]["location"]["latitude"], 
                      "long": result["property"]["location"]["longitude"],
                      "locality_precise" : result["property"]["location"]["locality"],
                      "type" : result["property"]["type"],
                      "province" : result["property"]["location"]["province"]
                      } for result in data['results']]
            return links
        else:
            print(f"Failed to fetch {url}: {response.status_code}")


