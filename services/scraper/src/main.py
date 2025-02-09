import os
from datetime import datetime
from typing import List, Dict
import requests
import pandas as pd
from multiprocessing.pool import ThreadPool
import json
from pathlib import Path
import sys
import math

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.get_house_data_scraper import get_house_data
from utils.get_links_data_scraper import get_links_from_page

# Configuration for different property types and their search parameters
PROPERTY_TYPES = {
    "house": {
        "url_templates": [
            "https://www.immoweb.be/fr/search-results/maison/a-vendre?buildingConditions=GOOD,AS_NEW,JUST_RENOVATED,TO_RENOVATE,TO_RESTORE,TO_BE_DONE_UP&propertyTypes=HOUSE&countries=BE&isALifeAnnuitySale=false&isAPublicSale=false&isNewlyBuilt=false&maxPrice=310000&minPrice=30000&minBedroomCount=1&minSurface=1&page={}&orderBy=relevance",
            # "https://www.immoweb.be/fr/search-results/maison/a-vendre?buildingConditions=GOOD,AS_NEW,JUST_RENOVATED,TO_RENOVATE,TO_RESTORE,TO_BE_DONE_UP&propertyTypes=HOUSE&countries=BE&isALifeAnnuitySale=false&isAPublicSale=false&isNewlyBuilt=false&maxPrice=550000&minPrice=311000&minBedroomCount=1&minSurface=1&page={}&orderBy=relevance",
            # "https://www.immoweb.be/fr/search-results/maison/a-vendre?buildingConditions=GOOD,AS_NEW,JUST_RENOVATED,TO_RENOVATE,TO_RESTORE,TO_BE_DONE_UP&propertyTypes=HOUSE&countries=BE&isALifeAnnuitySale=false&isAPublicSale=false&isNewlyBuilt=false&maxPrice=1500000&minPrice=551000&minBedroomCount=1&minSurface=1&page={}&orderBy=relevance",
        ],
        "pages": [10],
        # "pages": [334, 334, 189],
    },
    "apartment": {
        "url_templates": [
            "https://www.immoweb.be/fr/search-results/appartement/a-vendre?buildingConditions=GOOD,AS_NEW,JUST_RENOVATED,TO_RENOVATE,TO_RESTORE,TO_BE_DONE_UP&propertyTypes=APARTMENT&countries=BE&isALifeAnnuitySale=false&isAPublicSale=false&isNewlyBuilt=false&maxPrice=310000&minPrice=30000&minBedroomCount=1&minSurface=1&page={}&orderBy=relevance",
            # "https://www.immoweb.be/fr/search-results/appartement/a-vendre?buildingConditions=GOOD,AS_NEW,JUST_RENOVATED,TO_RENOVATE,TO_RESTORE,TO_BE_DONE_UP&propertyTypes=APARTMENT&countries=BE&isALifeAnnuitySale=false&isAPublicSale=false&isNewlyBuilt=false&maxPrice=1200000&minPrice=311000&minBedroomCount=1&minSurface=1&page={}&orderBy=relevance",
        ],
        "pages": [10],
        # "pages": [334, 200],
    },
}


class PropertyDataHandler:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define threshold for considering coordinates as "same location"
        # 0.0001 degrees is approximately 11 meters
        self.coordinate_threshold = 0.0001
        self.column_types = {
            "id": str,
            "zip_code": str,
            "price": float,
            "nb_bedrooms": "Int64",
            "living_area": "Int64",
            "surface_of_the_plot": "Int64",
            "nb_facades": "Int64",
            "garden_surface": "Int64",
            "terrace_surface": "Int64",
            "latitude": float,
            "longitude": float,
        }

    def haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points on Earth using Haversine formula"""
        R = 6371  # Earth's radius in kilometers

        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    def are_same_location(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> bool:
        """
        Determine if two sets of coordinates represent the same location.
        Uses Haversine distance and validates coordinate ranges.
        """
        # Validate coordinates
        if any(coord is None or coord == "" for coord in [lat1, lon1, lat2, lon2]):
            return False

        try:
            # Convert to float
            lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])

            # Validate ranges
            if not all(-90 <= lat <= 90 for lat in [lat1, lat2]):
                return False
            if not all(-180 <= lon <= 180 for lon in [lon1, lon2]):
                return False

            # Calculate actual distance in kilometers
            distance = self.haversine_distance(lat1, lon1, lat2, lon2)

            # Properties closer than 100 meters are considered same location
            return distance < 0.1

        except (ValueError, TypeError):
            return False

    def save_data(self, new_data: List[Dict], property_type: str) -> Dict[str, str]:
        """
        Save property data while handling duplicates based on location

        This function:
        1. Loads existing data if available
        2. Identifies duplicates using latitude/longitude
        3. Updates existing entries with new information when duplicates found
        4. Adds truly new properties to the dataset
        5. Maintains versioned backups for data safety
        """
        # Set up file paths
        main_file = os.path.join(self.output_dir, f"{property_type}_main.csv")
        backup_file = os.path.join(
            self.output_dir, f"{property_type}_backup_{self.timestamp}.csv"
        )

        # Convert new data to DataFrame
        # Convert new data to DataFrame with proper types
        new_df = pd.DataFrame(new_data)
        for col, dtype in self.column_types.items():
            if col in new_df.columns:
                try:
                    new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
                    new_df[col] = new_df[col].astype(dtype)
                except:
                    continue

        if os.path.exists(main_file):
            # Load existing data
            existing_df = pd.read_csv(main_file)

            # Convert existing data to proper types
            for col, dtype in self.column_types.items():
                if col in existing_df.columns:
                    try:
                        existing_df[col] = pd.to_numeric(
                            existing_df[col], errors="coerce"
                        )
                        existing_df[col] = existing_df[col].astype(dtype)
                    except:
                        continue

            # Create backup of current state
            existing_df.to_csv(backup_file, index=False)

            # Initialize a column to track duplicates
            new_df["is_duplicate"] = False

            # For each new property, check if it exists in the database
            for idx, new_row in new_df.iterrows():
                # Find potential matches based on location
                matches = existing_df.apply(
                    lambda x: self.are_same_location(
                        new_row["latitude"],
                        new_row["longitude"],
                        x["latitude"],
                        x["longitude"],
                    ),
                    axis=1,
                )

                if matches.any():
                    # Mark as duplicate
                    new_df.at[idx, "is_duplicate"] = True

                    # Get the index of the matching property
                    match_idx = matches[matches].index[0]

                    # Update the existing entry with new information
                    # This ensures we have the most recent data
                    for col in existing_df.columns:
                        try:
                            existing_df.loc[match_idx, col] = new_row[col]
                        except:
                            continue

            # Add only the non-duplicate properties
            truly_new = new_df[~new_df["is_duplicate"]].drop("is_duplicate", axis=1)
            updated_df = pd.concat([existing_df, truly_new], ignore_index=True)

            # Log the changes
            print(f"Found {new_df['is_duplicate'].sum()} duplicate properties")
            print(f"Adding {len(truly_new)} new properties")

        else:
            # If no existing file, use all new data
            updated_df = new_df
            print(f"Creating new database with {len(new_df)} properties")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Save the updated dataset
        updated_df.to_csv(main_file, index=False)

        return {
            "main_file": main_file,
            "backup_file": None,
            "new_entries": len(truly_new) if "truly_new" in locals() else len(new_df),
            "updated_entries": (
                new_df["is_duplicate"].sum()
                if "new_df" in locals() and "is_duplicate" in new_df
                else 0
            ),
        }


class PropertyScraper:
    """Main scraper class that coordinates the scraping process"""

    def __init__(self, output_dir: str = "data/raw", config_path: str = "config.json"):
        # Load configuration and initialize session
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
        with open(config_path, "r") as file:
            self.config = json.load(file)

        self.session = requests.Session()
        self.session.cookies.update(self.config["cookies"])
        self.session.headers.update(self.config["headers"])

        self.output_dir = output_dir
        self.data_handler = PropertyDataHandler(output_dir=output_dir)

    def scrape_property_type(self, property_type: str) -> str:
        """
        Scrape all properties of a specific type

        Args:
            property_type: Type of property to scrape (house/apartment)

        Returns:
            str: Path to the saved data file
        """
        url_config = PROPERTY_TYPES[property_type]
        all_pages = self._generate_pages(url_config)

        # Use ThreadPool instead of Pool
        print(f"Scraping links for {property_type}...")
        with ThreadPool() as pool:  # Changed this line
            all_links = pool.starmap(get_links_from_page, all_pages)
            flat_links = [link for sub_list in all_links for link in sub_list if link]

        print(f"Scraping details for {len(flat_links)} {property_type} listings...")
        urls = [(url, self.session) for url in flat_links if url["link"] is not None]

        # Use ThreadPool for the second parallel operation too
        with ThreadPool() as pool:  # Changed this line
            results = [
                res for res in pool.starmap(get_house_data, urls) if res is not None
            ]

        return self.data_handler.save_data(results, property_type)

    def _generate_pages(self, url_config: Dict) -> List:
        """Generate all pages to scrape based on configuration"""
        pages = []
        for url_template, page_count in zip(
            url_config["url_templates"], url_config["pages"]
        ):
            pages.extend(
                [
                    (url_template.format(page), self.session)
                    for page in range(page_count)
                ]
            )
        return pages


if __name__ == "__main__":
    scraper = PropertyScraper()

    for property_type in PROPERTY_TYPES.keys():
        try:
            print(f"\nStarting scraping for {property_type}s...")
            result = scraper.scrape_property_type(property_type)

            print(f"\nScraping results for {property_type}s:")
            print(f"Main data file: {result['main_file']}")
            print(f"New entries added: {result['new_entries']}")
            print(f"Existing entries updated: {result['updated_entries']}")
            if result["backup_file"]:
                print(f"Backup created at: {result['backup_file']}")

        except Exception as e:
            print(f"Error scraping {property_type}s: {str(e)}")
