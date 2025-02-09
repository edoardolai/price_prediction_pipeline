import re
from bs4 import BeautifulSoup as bs
import urllib.parse
from requests import Session
from typing import Dict, Optional


def get_house_data(url: str, session: Session) -> Optional[Dict[str, Optional[object]]]:
    """
    Scrapes real estate data from a given property URL.

        Args:
            url (str): The URL of the property listing.
            session (Session): A `requests.Session` object for managing HTTP connections and cookies.

        Returns:
            Optional[Dict[str, Optional[object]]]: A dictionary containing property details such as locality,
            zip code, property type, price, number of bedrooms, living area, and other amenities.
            Returns `None` if the request fails or data extraction is incomplete.

        Raises:
            ValueError: If the URL format does not match the expected pattern for key data extraction.
    """
    print("url is", url["link"].strip("\n"))
    response = session.get(url["link"].strip("\n"))
    house_dict = dict()
    if response.status_code == 200:
        house_page = bs(response.content, "html.parser")
        cleaned_url = re.sub(
            r"[^\w\s()\u00C0-\u017F-]/+|[\s']*", "", urllib.parse.unquote(url["link"])
        )
        re.findall(r"for-sale/(\w+([-\w*])*)", urllib.parse.unquote(cleaned_url))
        district_match = re.findall(r"for-sale/(\w+([-\w*])*)", cleaned_url)
        # Not all links have these properties so sometimes an empty list is returned, causing an out of range index error
        house_dict["district"] = (
            district_match[0][0].title() if district_match else None
        )
        house_dict["locality"] = url["locality_precise"]
        house_dict["latitude"] = url["lat"]
        house_dict["longitude"] = url["long"]
        house_dict["provice"] = url["province"]
        id_match = re.findall(r"/(\d{4})/(\d+)/", cleaned_url)
        house_dict["id"] = id_match[0][1].title() if id_match else None
        zip_match = re.findall(r"/(\d{4})/", cleaned_url)
        house_dict["zip_code"] = zip_match[0].title() if zip_match else None
        property_sub_type_match = re.findall(r"(classified)/(\w+[_\w*]*)", cleaned_url)
        house_dict["property_sub_type"] = (
            property_sub_type_match[0][1].title() if property_sub_type_match else None
        )
        house_dict["property_type"] = url["type"]
        price = (
            house_page.select_one(".classified__price .sr-only").get_text().strip("€")
        )
        house_dict["price"] = price if price != "" else None
        house_dict["nb_bedrooms"] = extract_table_data(house_page, r"Bedrooms")
        house_dict["living_area"] = extract_table_data(house_page, r"Living\sarea")
        house_dict["surface_of_the_plot"] = extract_table_data(
            house_page, r"Surface\sof\sthe\splot"
        )
        house_dict["nb_facades"] = extract_table_data(
            house_page, r"Number\sof\sfrontages"
        )
        house_dict["state_of_building"] = extract_table_data(
            house_page, r"Building\scondition"
        )
        fireplace = extract_table_data(house_page, r"How\smany\sfireplaces")
        fireplace = 1 if fireplace is not None and int(fireplace) > 0 else 0
        house_dict["fireplace"] = fireplace
        kitchen_type = extract_table_data(house_page, r"Kitchen\stype")
        kitchen_type_list = [
            "Installed",
            "Hyper equipped",
            "USA installed",
            "USA hyper equipped",
        ]
        kitchen_type = 1 if kitchen_type in kitchen_type_list else 0
        house_dict["equipped_kitchen"] = kitchen_type
        garden_surface = extract_table_data(house_page, r"Garden\ssurface")
        garden, garden_surface = (
            (0, None) if garden_surface is None else (1, garden_surface)
        )
        house_dict["garden"] = garden
        house_dict["garden_surface"] = garden_surface
        terrace_surface = extract_table_data(house_page, r"Terrace\ssurface")
        terrace, terrace_surface = (
            (0, None) if terrace_surface is None else (1, terrace_surface)
        )
        house_dict["terrace"] = terrace
        house_dict["terrace_surface"] = terrace_surface
        furnished = extract_table_data(house_page, r"Furnished")
        furnished = 1 if furnished == "Yes" else 0
        house_dict["furnished"] = furnished
        swimming_pool = extract_table_data(house_page, r"Swimming\spool")
        swimming_pool = 1 if swimming_pool == "Yes" else 0
        house_dict["swimming_pool"] = swimming_pool
        return house_dict
    else:
        print(f"Failed to fetch {url}: {response.status_code}")
        return None


def extract_table_data(page: bs, regex: str) -> Optional[str]:
    """
    Extracts data from a table cell matching a given regex pattern.

    Args:
        page (bs): The BeautifulSoup object representing the page content.
        regex (str): The regular expression pattern to search for in table headers.

    Returns:
        Optional[str]: The text content of the corresponding table cell if found, otherwise None.
    """
    searched_tag = page.find("th", string=re.compile(regex))
    if searched_tag is not None:
        return page.find(
            "th", string=re.compile(regex)
        ).next_sibling.next_element.next_element.strip()
    else:
        return None
