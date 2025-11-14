import json
import os

import pandas as pd


def parse_jerusalem_streets(geojson_file):
    """
    Parse GeoJSON file from Overpass Turbo containing Jerusalem streets
    and extract Hebrew and English street names.

    Args:
        geojson_file (str): Path to the GeoJSON file

    Returns:
        pandas.DataFrame: DataFrame containing street names in Hebrew and English
    """
    # Read the GeoJSON file
    with open(geojson_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract street names
    streets = []

    for element in data['features']:
        if 'properties' in element and 'name:en' in element['properties'] and 'name:he' in element['properties']:
            street_info = {
                'id': element['id'],
                'type': element['type'],
                'name_en': element['properties'].get('name:en', ''),
                'name_he': element['properties'].get('name:he', '')
            }

            # Add additional information if available
            if 'highway' in element['properties']:
                street_info['highway_type'] = element['properties']['highway']

            streets.append(street_info)

    # Convert to DataFrame
    df = pd.DataFrame(streets)

    # Remove duplicates based on both names
    df = df.drop_duplicates(subset=['name_en', 'name_he'])

    return df


def save_as_csv(df, output_file="../../output/geolocating/jerusalem_streets.csv"):
    """Save DataFrame to CSV file"""
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} unique streets to {output_file}")


def convert_to_dict(df):
    """Convert DataFrame to dictionary format"""
    streets_dict = {}

    for _, row in df.iterrows():
        streets_dict[row['id']] = {
            'name_en': row['name_en'],
            'name_he': row['name_he'],
            'type': row['type']
        }
        if 'highway_type' in row:
            streets_dict[row['id']]['highway_type'] = row['highway_type']

    return streets_dict


def create_street_names_db():
    # Replace with your GeoJSON file path
    geojson_file = "../../input/jerusalem_streets.geojson"
    if os.path.exists(geojson_file):
        # Parse the GeoJSON and create DataFrame
        streets_df = parse_jerusalem_streets(geojson_file)

        print(f"Found {len(streets_df)} unique streets in Jerusalem")

        # Display first few entries
        print("\nFirst 5 streets:")
        print(streets_df.head())

        # Save to CSV
        save_as_csv(streets_df)

        # Convert to dictionary if needed
        streets_dict = convert_to_dict(streets_df)
        print(f"\nDictionary created with {len(streets_dict)} entries")

        # Example of accessing dictionary
        if streets_dict:
            first_key = list(streets_dict.keys())[0]
            print(f"\nExample dictionary entry:\nID: {first_key}\nData: {streets_dict[first_key]}")
    else:
        print(f"Error: File '{geojson_file}' not found.")
        print(
            "Please save your Overpass Turbo GeoJSON result as 'jerusalem_streets.geojson' in the same directory as this script.")


def parse_jerusalem_neighborhoods(geojson_file):
    """
    Parse GeoJSON file from Overpass Turbo containing Jerusalem neighborhoods
    and extract Hebrew and English neighborhood names.

    Args:
        geojson_file (str): Path to the GeoJSON file

    Returns:
        pandas.DataFrame: DataFrame containing neighborhood names in Hebrew and English
    """
    # Read the GeoJSON file
    with open(geojson_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    neighborhoods = []

    for feature in data['features']:
        props = feature.get('properties', {})
        if 'name:en' in props and 'name:he' in props:
            neighborhoods.append({
                'id': feature.get('id'),
                'type': feature.get('geometry', {}).get('type', ''),
                'name_en': props.get('name:en', ''),
                'name_he': props.get('name:he', ''),
                'place_type': props.get('place', '')
            })

    df = pd.DataFrame(neighborhoods)
    df = df.drop_duplicates(subset=['name_en', 'name_he'])

    return df

def save_neighborhoods_csv(df, output_file="../../output/geolocating/jerusalem_neighborhoods.csv"):
    """Save neighborhood DataFrame to CSV file"""
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} unique neighborhoods to {output_file}")

def convert_neighborhoods_to_dict(df):
    """Convert neighborhood DataFrame to dictionary format"""
    return {
        row['id']: {
            'name_en': row['name_en'],
            'name_he': row['name_he'],
            'type': row['type'],
            'place_type': row.get('place_type', '')
        }
        for _, row in df.iterrows()
    }

def create_neighborhood_names_db():
    geojson_file = "../../input/jerusalem_neighborhoods.geojson"

    if os.path.exists(geojson_file):
        df = parse_jerusalem_neighborhoods(geojson_file)
        print(f"Found {len(df)} unique neighborhoods in Jerusalem")

        print("\nFirst 5 neighborhoods:")
        print(df.head())

        save_neighborhoods_csv(df)

        neighborhood_dict = convert_neighborhoods_to_dict(df)
        print(f"\nDictionary created with {len(neighborhood_dict)} entries")

        if neighborhood_dict:
            first_key = list(neighborhood_dict.keys())[0]
            print(f"\nExample dictionary entry:\nID: {first_key}\nData: {neighborhood_dict[first_key]}")
    else:
        print(f"Error: File '{geojson_file}' not found.")
        print("Please save your Overpass Turbo GeoJSON result as 'jerusalem_neighborhoods.geojson' in the '../input' folder.")



import json

# Input file path - Updated to use the municipal file
INPUT_GEOJSON = "../../input/jerusalem_neighborhoods_from_munic.geojson"

# Output file paths
HAREDI_OUTPUT = "../../input/jerusalem_haredi_neighborhoods.geojson"
ARAB_OUTPUT = "../../input/jerusalem_arab_neighborhoods.geojson"
GOV_OUTPUT =  "../../input/jerusalem_government_neighborhoods.geojson"

# Neighborhood names in Hebrew (as they appear in SCHN_NAME field)
haredi_names = [
    "גאולה",
    "מאה שערים",
    "סנהדריה",
    "בית ישראל",
    "רוממה",
    "רמת שלמה",
    "נוה יעקב",
    "גבעת שאול",
    "בית וגן",
    "רמות",
    "שמואל הנביא",
    "תל ארזה, עזרת",
    "שמעון הצדיק",
    "הבוכרים",
    "קריית משה"
]

arab_names = [
    "שועפאט",
    "בית חנינא",
    "עיסאוויה",
    "צור באהר",
    "א-טור",
    "סילואן",
    "ואדי ג'וז",
    "אום טובא",
    "אום ליסון",
    "ראס אל עמוד",
    "א-רם",
    "ג'בל מוקאבר",
    "באב א זהרה",
    "א-שייאח",
    "א-סוואנה",
    'בית צפפא,שרפת' ,
    'מחנה שועפט',
    'שייח גר\'אח',
    'ערב א-סוואחרה',
    'כפר עקב'
]

gov_univ_industrial_names = [
    "אזור תעשיה גבעת שאול",
    "אזור תעשיה עטרות",
    "גבעת רם",
    "הר הרצל",
    "יד ושם - הר הזיכרון",
    "קריית הלאום",
    "קריית הממשלה",
    "תלפיות - תעשיה"
]


def filter_features_by_schn_name(features, names_list):
    """Filter features based on SCHN_NAME field instead of name:en"""
    return [
        feature for feature in features
        if feature.get("properties", {}).get("SCHN_NAME") in names_list
    ]

def create_neighborhoods_files():
    # Load the input GeoJSON
    with open(INPUT_GEOJSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract features
    features = data.get("features", [])

    # Filter features using SCHN_NAME field
    haredi_features = filter_features_by_schn_name(features, haredi_names)
    arab_features = filter_features_by_schn_name(features, arab_names)
    gov_features = filter_features_by_schn_name(features, gov_univ_industrial_names)
    # Save new GeoJSONs
    for filename, filtered in [(HAREDI_OUTPUT, haredi_features), (ARAB_OUTPUT, arab_features), (GOV_OUTPUT, gov_features)]:
        output = {
            "type": "FeatureCollection",
            "features": filtered
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(filtered)} features to {filename}")


if __name__ == "__main__":

    #create_street_names_db()
    #create_neighborhood_names_db()
    create_neighborhoods_files()