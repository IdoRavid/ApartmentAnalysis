import json
import os
import random
import re
from collections import Counter
from shapely.geometry import shape
import pandas as pd
import matplotlib.pyplot as plt
import pyproj
from tqdm import tqdm
from fuzzywuzzy import fuzz, process
from flashtext import KeywordProcessor
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
# Add tqdm support for pandas apply
tqdm.pandas()


class OptimizedMatchingPreferences:
    """Enhanced configuration based on validation analysis"""

    def __init__(self):
        # Strategy remains the same
        self.strategy = "neighborhood_first"

        # CRITICAL FIX: Raise minimum confidence threshold
        # Your 80-90% range has only 62.5% accuracy
        self.min_confidence_threshold = 95  # Increased from 70

        # Tighten fuzzy matching (11/15 errors are fuzzy matches)
        self.fuzzy_threshold = 95  # Increased from 85
        self.min_fuzzy_confidence = 90  # Increased from 80

        # Keep exact match confidence high
        self.exact_neighborhood_confidence = 100
        self.exact_street_confidence = 95  # Slightly increased

        # Enhanced exclusion for problematic names identified in validation
        self.problematic_fuzzy_matches = {
            "אחד העם", "שחר", "מוריה", "הארז", "לחי" # From your incorrect samples
        }

        # Stricter length requirements
        self.min_street_name_length = 4  # Was 2
        self.min_neighborhood_name_length = 2  # Was 2

        # Popular locations (unchanged)
        self.overly_popular_neighborhoods = {
            "רחביה", "נחלאות", "גבעת רם", "הר הצופים",
            "ארנונה", "בית הכרם", "עין כרם", "ניות"
        }

        self.overly_popular_streets = {
            "יפו", "בצלאל", "עזה"
        }


# Enhanced validation functions
def is_valid_street_name_optimized(name, preferences):
    """Enhanced street name validation with stricter requirements"""
    if pd.isna(name) or not name:
        return False

    cleaned_name = clean_name(name).strip()

    # Stricter length requirement
    if len(cleaned_name) < preferences.min_street_name_length:
        return False

    # Enhanced exclusion list based on validation results
    excluded_names = {
        "דר", "שי", "רות", "גד", "שבו", "חבר", "גבע", "המרכז", "ליש",
        "שלום רב", "דוד", "הנציב", "מעש", "בדאיה", "פרס", "נאה", "אדם",
        "חגי", "מחל", "כי טוב", "עובד", "הרכבת", "האוניברסיטה העברית",
        "ניות", "יפו", "נרות שבת", "הרכב", "מנחם", "רמה", "גבעה", "עמק",
        "הר", "נחל", "שמיר", "בית הכרם", "כביש 57", "לחי", "דביר"
    }

    return cleaned_name not in excluded_names


def is_valid_neighborhood_name_optimized(name, preferences):
    """Enhanced neighborhood name validation with stricter requirements"""
    if pd.isna(name) or not name:
        return False

    cleaned_name = clean_name(name).strip()

    # Stricter length requirement
    if len(cleaned_name) < preferences.min_neighborhood_name_length:
        return False

    # Enhanced exclusion list based on validation results
    excluded_neighborhoods = {
        "דר", "שי", "רות", "גד", "שבו", "חבר", "גבע", "המרכז", "ליש",
        "שלום רב", "דוד", "הנציב", "מעש", "בדאיה", "פרס", "נאה", "אדם",
        "חגי", "האוניברסיטה העברית",
    }

    return cleaned_name not in excluded_neighborhoods


def is_problematic_fuzzy_match(name, preferences):
    """Check if this is a known problematic fuzzy match"""
    cleaned_name = clean_name(name).strip()
    return cleaned_name in preferences.problematic_fuzzy_matches


# Enhanced extraction function
def extract_best_location_optimized(text, street_kp, neighborhood_kp, streets_data, neighborhoods_data, preferences):
    """
    Optimized extraction with enhanced validation but without context layer
    """
    if not text or pd.isna(text):
        return None

    all_matches = []
    found_popular_neighborhood = False
    found_popular_street = False
    popular_neighborhood_matches = []
    popular_street_matches = []

    # 1. Neighborhood matches with enhanced validation
    exact_neighborhood_matches = neighborhood_kp.extract_keywords(text)
    if exact_neighborhood_matches:
        for neighborhood_id, original_name in exact_neighborhood_matches:
            if is_valid_neighborhood_name_optimized(original_name, preferences):
                is_popular = clean_name(original_name).strip() in preferences.overly_popular_neighborhoods

                match = {
                    'id': neighborhood_id,
                    'name': original_name,
                    'confidence': preferences.exact_neighborhood_confidence,
                    'priority': 1,
                    'type': 'neighborhood',
                    'method': 'exact',
                    'is_popular': is_popular
                }

                if is_popular:
                    found_popular_neighborhood = True
                    popular_neighborhood_matches.append(match)
                else:
                    all_matches.append(match)

    # 2. Street matches with enhanced validation
    exact_street_matches = street_kp.extract_keywords(text)
    street_matches = []
    if exact_street_matches:
        for street_id, original_name, highway_type in exact_street_matches:
            if is_valid_street_name_optimized(original_name, preferences):
                confidence = preferences.exact_street_confidence
                # Only include high-confidence exact street matches
                if confidence >= 90:
                    is_popular = clean_name(original_name).strip() in preferences.overly_popular_streets

                    match = {
                        'id': street_id,
                        'name': original_name,
                        'confidence': confidence,
                        'priority': 2 + get_highway_priority(highway_type),
                        'type': 'street',
                        'method': 'exact',
                        'is_popular': is_popular
                    }

                    if is_popular:
                        found_popular_street = True
                        popular_street_matches.append(match)
                    else:
                        street_matches.append(match)

    # 3. ENHANCED fuzzy matches with strict validation
    fuzzy_matches = process.extractBests(
        text,
        streets_data['search_strings'],
        scorer=fuzz.partial_ratio,
        score_cutoff=preferences.fuzzy_threshold,
        limit=3  # Reduced from 5 to focus on best matches
    )

    for match_text, fuzzy_score in fuzzy_matches:
        if fuzzy_score >= preferences.min_fuzzy_confidence:
            street_info = streets_data['lookup'][match_text]
            street_id, original_name, highway_type = street_info

            # Enhanced validation for fuzzy matches
            if (is_valid_street_name_optimized(original_name, preferences) and
                    not is_problematic_fuzzy_match(original_name, preferences) and
                    not any(m['id'] == street_id for m in street_matches) and
                    not any(m['id'] == street_id for m in popular_street_matches)):

                # Require higher confidence for very short matches
                if len(clean_name(original_name).strip()) <= 3 and fuzzy_score < 95:
                    continue

                is_popular = clean_name(original_name).strip() in preferences.overly_popular_streets

                match = {
                    'id': street_id,
                    'name': original_name,
                    'confidence': fuzzy_score,
                    'priority': 10 + get_highway_priority(highway_type),
                    'type': 'street',
                    'method': 'fuzzy',
                    'is_popular': is_popular
                }

                if is_popular:
                    found_popular_street = True
                    popular_street_matches.append(match)
                else:
                    street_matches.append(match)

    # Handle combinations of popular and non-popular matches (same logic as before)
    if found_popular_neighborhood and not found_popular_street and street_matches:
        for street_match in street_matches:
            street_match['priority'] = 0.5
            all_matches.append(street_match)
        for pop_match in popular_neighborhood_matches:
            pop_match['priority'] = 1.5
            all_matches.append(pop_match)
    elif found_popular_street and not found_popular_neighborhood and all_matches:
        for pop_street_match in popular_street_matches:
            pop_street_match['priority'] = 2.5
            all_matches.append(pop_street_match)
    elif found_popular_neighborhood and found_popular_street:
        for pop_neighborhood_match in popular_neighborhood_matches:
            pop_neighborhood_match['priority'] = 1.0
            all_matches.append(pop_neighborhood_match)
        for pop_street_match in popular_street_matches:
            pop_street_match['priority'] = 2.0
            all_matches.append(pop_street_match)
    else:
        all_matches.extend(street_matches)
        all_matches.extend(popular_neighborhood_matches)
        all_matches.extend(popular_street_matches)

    # Filter by minimum confidence (now 90%)
    all_matches = [m for m in all_matches if m['confidence'] >= preferences.min_confidence_threshold]

    if not all_matches:
        return None

    # Sort by priority, then confidence
    all_matches.sort(key=lambda x: (x['priority'], -x['confidence']))

    return all_matches[0]

def normalize_hebrew(text):
    """Clean Hebrew text by removing diacritics and normalizing spaces"""
    if pd.isna(text):
        return ""
    text = re.sub(r"[\u0591-\u05C7]", "", str(text))
    return re.sub(r"\s+", " ", text.strip().lower())


def clean_name(name):
    """Remove non-letter characters (except spaces) from names"""
    if pd.isna(name):
        return ""
    cleaned = re.sub(r'[^\u0590-\u05FF\u0041-\u005A\u0061-\u007A\s]', '', str(name))
    return cleaned.strip()


def is_valid_street_name(name):
    """Check if street name should be excluded from matching - updated exclusion list"""
    if pd.isna(name) or not name:
        return False

    cleaned_name = clean_name(name).strip()

    if len(cleaned_name) <= 2:
        return False

    # Enhanced exclusion list based on validation results
    excluded_names = {
        "דר", "שי", "רות", "גד", "שבו", "חבר", "גבע", "המרכז", "ליש",
        "שלום רב", "דוד", "הנציב", "מעש", "בדאיה", "פרס", "נאה", "אדם",
        "חגי", "מחל", "כי טוב", "עובד", "הרכבת", "האוניברסיטה העברית",
        "ניות", "יפו", "נרות שבת", "הרכב", "מנחם", "רמה", "גבעה", "עמק", "הר", "נחל",
        "שמיר", "בית הכרם"
    }

    return cleaned_name not in excluded_names


def is_valid_neighborhood_name(name):
    """Check if neighborhood name should be excluded from matching - updated exclusion list"""
    if pd.isna(name) or not name:
        return False

    cleaned_name = clean_name(name).strip()

    if len(cleaned_name) <= 2:
        return False

    # Enhanced exclusion list based on validation results
    excluded_neighborhoods = {
        "דר", "שי", "רות", "גד", "שבו", "חבר", "גבע", "המרכז", "ליש",
        "שלום רב", "דוד", "הנציב", "מעש", "בדאיה", "פרס", "נאה", "אדם",
        "חגי", "האוניברסיטה העברית",
    }

    return cleaned_name not in excluded_neighborhoods


def get_highway_priority(highway_type):
    """Assign priority scores - lower scores = higher priority (prefer smaller streets)"""
    priority_map = {
        'residential': 1, 'living_street': 1, 'service': 2, 'unclassified': 3,
        'secondary': 4, 'tertiary': 5, 'primary': 6, 'trunk': 7, 'motorway': 8,
        'motorway_link': 8, 'trunk_link': 7, 'primary_link': 6,
        'secondary_link': 5, 'tertiary_link': 5, 'unknown': 10
    }
    return priority_map.get(str(highway_type).lower(), 10)


def extract_best_location(text, street_kp, neighborhood_kp, streets_data, neighborhoods_data, preferences):
    """
    Extract the single best location match using improved strategy with popular street/neighborhood handling
    Returns: dict with best match info or None
    """
    if not text or pd.isna(text):
        return None

    all_matches = []
    found_popular_neighborhood = False
    found_popular_street = False
    popular_neighborhood_matches = []
    popular_street_matches = []

    # 1. Extract neighborhood matches (highest priority, but check for popular ones)
    exact_neighborhood_matches = neighborhood_kp.extract_keywords(text)
    if exact_neighborhood_matches:
        for neighborhood_id, original_name in exact_neighborhood_matches:
            if is_valid_neighborhood_name(original_name):
                # Check if this is an overly popular neighborhood
                is_popular = clean_name(original_name).strip() in preferences.overly_popular_neighborhoods

                match = {
                    'id': neighborhood_id,
                    'name': original_name,
                    'confidence': preferences.exact_neighborhood_confidence,
                    'priority': 1,  # Highest priority normally
                    'type': 'neighborhood',
                    'method': 'exact',
                    'is_popular': is_popular
                }

                if is_popular:
                    found_popular_neighborhood = True
                    popular_neighborhood_matches.append(match)
                else:
                    all_matches.append(match)

    # 2. Extract exact street matches (second priority, but check for popular ones)
    exact_street_matches = street_kp.extract_keywords(text)
    street_matches = []
    if exact_street_matches:
        for street_id, original_name, highway_type in exact_street_matches:
            if is_valid_street_name(original_name):
                confidence = preferences.exact_street_confidence
                # Only include high-confidence exact street matches
                if confidence >= 90:
                    # Check if this is an overly popular street
                    is_popular = clean_name(original_name).strip() in preferences.overly_popular_streets

                    match = {
                        'id': street_id,
                        'name': original_name,
                        'confidence': confidence,
                        'priority': 2 + get_highway_priority(highway_type),
                        'type': 'street',
                        'method': 'exact',
                        'is_popular': is_popular
                    }

                    if is_popular:
                        found_popular_street = True
                        popular_street_matches.append(match)
                    else:
                        street_matches.append(match)

    # 3. Extract fuzzy street matches (lowest priority, strict filtering)
    fuzzy_matches = process.extractBests(
        text,
        streets_data['search_strings'],
        scorer=fuzz.partial_ratio,
        score_cutoff=preferences.fuzzy_threshold,
        limit=5
    )

    for match_text, fuzzy_score in fuzzy_matches:
        if fuzzy_score >= preferences.min_fuzzy_confidence:
            street_info = streets_data['lookup'][match_text]
            street_id, original_name, highway_type = street_info

            if (is_valid_street_name(original_name) and
                    not any(m['id'] == street_id for m in street_matches) and
                    not any(m['id'] == street_id for m in popular_street_matches)):

                # Check if this is an overly popular street
                is_popular = clean_name(original_name).strip() in preferences.overly_popular_streets

                match = {
                    'id': street_id,
                    'name': original_name,
                    'confidence': fuzzy_score,
                    'priority': 10 + get_highway_priority(highway_type),
                    'type': 'street',
                    'method': 'fuzzy',
                    'is_popular': is_popular
                }

                if is_popular:
                    found_popular_street = True
                    popular_street_matches.append(match)
                else:
                    street_matches.append(match)

    # Handle combinations of popular and non-popular matches
    if found_popular_neighborhood and not found_popular_street and street_matches:
        # Popular neighborhood + regular street: prefer the street
        for street_match in street_matches:
            street_match['priority'] = 0.5  # Higher priority than neighborhood (1)
            all_matches.append(street_match)
        # Add popular neighborhoods with lower priority
        for pop_match in popular_neighborhood_matches:
            pop_match['priority'] = 1.5
            all_matches.append(pop_match)

    elif found_popular_street and not found_popular_neighborhood and all_matches:
        # Popular street + regular neighborhood: prefer the neighborhood
        # Regular neighborhoods are already in all_matches with priority 1
        # Add popular streets with lower priority
        for pop_street_match in popular_street_matches:
            pop_street_match['priority'] = 2.5  # Lower priority than neighborhood (1)
            all_matches.append(pop_street_match)

    elif found_popular_neighborhood and found_popular_street:
        # Both popular: prefer neighborhood (original strategy)
        for pop_neighborhood_match in popular_neighborhood_matches:
            pop_neighborhood_match['priority'] = 1.0
            all_matches.append(pop_neighborhood_match)
        for pop_street_match in popular_street_matches:
            pop_street_match['priority'] = 2.0
            all_matches.append(pop_street_match)

    else:
        # Normal behavior: add all matches with their original priorities
        all_matches.extend(street_matches)
        all_matches.extend(popular_neighborhood_matches)
        all_matches.extend(popular_street_matches)

    # Filter by minimum confidence threshold
    all_matches = [m for m in all_matches if m['confidence'] >= preferences.min_confidence_threshold]

    if not all_matches:
        return None

    # Sort by priority (lower = better), then by confidence (higher = better)
    all_matches.sort(key=lambda x: (x['priority'], -x['confidence']))

    return all_matches[0]


def load_and_prepare_data():
    """Load and prepare streets and neighborhoods data"""
    print("Loading and preparing data...")

    # Load and clean streets
    streets_df = pd.read_csv("../../output/geolocating/jerusalem_streets.csv")
    streets_df = streets_df.dropna(subset=["name_he"])
    streets_df["cleaned_name"] = streets_df["name_he"].apply(clean_name)
    streets_df["normalized_name"] = streets_df["cleaned_name"].astype(str).map(normalize_hebrew)
    streets_df = streets_df[streets_df["cleaned_name"].apply(is_valid_street_name)]

    use_street_id = 'id' if 'id' in streets_df.columns else 'index'
    if use_street_id == 'index':
        streets_df = streets_df.reset_index()

    # Load and clean neighborhoods
    neighborhoods_df = pd.read_csv("../../output/geolocating/jerusalem_neighborhoods.csv")
    neighborhoods_df = neighborhoods_df.dropna(subset=["name_he"])
    neighborhoods_df["cleaned_name"] = neighborhoods_df["name_he"].apply(clean_name)
    neighborhoods_df["normalized_name"] = neighborhoods_df["cleaned_name"].astype(str).map(normalize_hebrew)
    neighborhoods_df = neighborhoods_df[neighborhoods_df["cleaned_name"].apply(is_valid_neighborhood_name)]

    use_neighborhood_id = 'id' if 'id' in neighborhoods_df.columns else 'index'
    if use_neighborhood_id == 'index':
        neighborhoods_df = neighborhoods_df.reset_index()

    print(f"Loaded {len(streets_df)} valid streets and {len(neighborhoods_df)} valid neighborhoods")

    return streets_df, neighborhoods_df, use_street_id, use_neighborhood_id


def build_matching_structures(streets_df, neighborhoods_df, use_street_id, use_neighborhood_id):
    """Build keyword processors and lookup structures"""
    print("Building matching structures...")

    street_kp = KeywordProcessor(case_sensitive=False)
    neighborhood_kp = KeywordProcessor(case_sensitive=False)

    streets_data = {'search_strings': [], 'lookup': {}}
    neighborhoods_data = {'search_strings': [], 'lookup': {}}

    # Build streets data
    for _, row in streets_df.iterrows():
        normalized = row["normalized_name"]
        original_name = row["name_he"]
        street_id = row[use_street_id]
        highway_type = row.get("highway", "unknown")

        street_kp.add_keyword(normalized, (street_id, original_name, highway_type))
        streets_data['search_strings'].append(normalized)
        streets_data['lookup'][normalized] = (street_id, original_name, highway_type)

    # Build neighborhoods data
    for _, row in neighborhoods_df.iterrows():
        normalized = row["normalized_name"]
        original_name = row["name_he"]
        neighborhood_id = row[use_neighborhood_id]

        neighborhood_kp.add_keyword(normalized, (neighborhood_id, original_name))
        neighborhoods_data['search_strings'].append(normalized)
        neighborhoods_data['lookup'][normalized] = (neighborhood_id, original_name)

    return street_kp, neighborhood_kp, streets_data, neighborhoods_data


def geolocate_posts_improved():
    """
    Main improved geolocation function with validation-based enhancements
    """
    preferences = OptimizedMatchingPreferences()

    print("=== IMPROVED GEOLOCATION SYSTEM ===")
    print(f"Strategy: {preferences.strategy}")
    print(f"Fuzzy threshold: {preferences.fuzzy_threshold}")
    print(f"Min confidence threshold: {preferences.min_confidence_threshold}")
    print(f"Min fuzzy confidence: {preferences.min_fuzzy_confidence}")
    print(f"Popular neighborhoods (prefer streets when co-occurring): {preferences.overly_popular_neighborhoods}")
    print(f"Popular streets (prefer neighborhoods when co-occurring): {preferences.overly_popular_streets}")
    print()

    # Load and prepare data
    streets_df, neighborhoods_df, use_street_id, use_neighborhood_id = load_and_prepare_data()

    # Build matching structures
    street_kp, neighborhood_kp, streets_data, neighborhoods_data = build_matching_structures(
        streets_df, neighborhoods_df, use_street_id, use_neighborhood_id
    )

    # Load posts
    print("Loading posts...")
    df = pd.read_csv("../../output/feature_extraction/PostsWithFeatures.csv")
    df["post_text"] = df["post_text"].astype(str).map(normalize_hebrew)

    print(f"Processing {len(df)} posts...")

    # Extract best location for each post
    results = df["post_text"].progress_apply(
        lambda x: extract_best_location(
            x, street_kp, neighborhood_kp, streets_data, neighborhoods_data, preferences
        )
    )

    # Extract information from results
    df["location_id"] = results.map(lambda x: x['id'] if x else None)
    df["location_name"] = results.map(lambda x: x['name'] if x else None)
    df["location_confidence"] = results.map(lambda x: x['confidence'] if x else None)
    df["location_type"] = results.map(lambda x: x['type'] if x else None)
    df["location_method"] = results.map(lambda x: x['method'] if x else None)

    # Statistics with popular street/neighborhood handling
    total_posts = len(df)
    posts_with_location = df["location_name"].notna().sum()
    coverage = posts_with_location / total_posts * 100

    # Check how often we chose different types over popular ones
    popular_neighborhoods_mentioned = df[df["location_name"].notna()].apply(
        lambda row: clean_name(row["location_name"]).strip() in preferences.overly_popular_neighborhoods
        if row["location_type"] == "neighborhood" else False, axis=1
    ).sum()

    popular_streets_mentioned = df[df["location_name"].notna()].apply(
        lambda row: clean_name(row["location_name"]).strip() in preferences.overly_popular_streets
        if row["location_type"] == "street" else False, axis=1
    ).sum()

    streets_chosen = df[df["location_type"] == "street"].shape[0]
    neighborhoods_chosen = df[df["location_type"] == "neighborhood"].shape[0]

    print(f"\n=== RESULTS ===")
    print(f"Total posts: {total_posts}")
    print(f"Posts with location: {posts_with_location}")
    print(f"Coverage: {coverage:.1f}%")
    print(f"Popular neighborhoods chosen: {popular_neighborhoods_mentioned}")
    print(f"Popular streets chosen: {popular_streets_mentioned}")
    print(f"Total streets chosen: {streets_chosen}")
    print(f"Total neighborhoods chosen: {neighborhoods_chosen}")

    # Show examples where we might have made smart choices
    streets_with_popular_neighborhood_context = 0
    neighborhoods_with_popular_street_context = 0

    for _, row in df[df["location_name"].notna()].iterrows():
        text = row["post_text"]

        if row["location_type"] == "street":
            # Check if street was chosen in context of popular neighborhood
            for pop_neighborhood in preferences.overly_popular_neighborhoods:
                if normalize_hebrew(pop_neighborhood) in text:
                    streets_with_popular_neighborhood_context += 1
                    break

        elif row["location_type"] == "neighborhood":
            # Check if neighborhood was chosen in context of popular street
            for pop_street in preferences.overly_popular_streets:
                if normalize_hebrew(pop_street) in text:
                    neighborhoods_with_popular_street_context += 1
                    break

    print(f"Streets chosen in context of popular neighborhoods: {streets_with_popular_neighborhood_context}")
    print(f"Neighborhoods chosen in context of popular streets: {neighborhoods_with_popular_street_context}")

    # Breakdown by type and method
    location_stats = df[df["location_name"].notna()].groupby(['location_type', 'location_method']).size()
    print(f"\nBreakdown by type and method:")
    for (type_name, method), count in location_stats.items():
        print(f"  {type_name} ({method}): {count}")

    # Confidence distribution
    confidence_stats = df[df["location_confidence"].notna()]["location_confidence"].describe()
    print(f"\nConfidence distribution:")
    print(f"  Mean: {confidence_stats['mean']:.1f}")
    print(f"  Min: {confidence_stats['min']:.1f}")
    print(f"  Max: {confidence_stats['max']:.1f}")

    # Save results
    output_path = "../../output/geolocating/posts_improved_geolocation.csv"
    df.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(f"\nResults saved to: {output_path}")

    return df





def plot_improved_results(csv_path="../../output/geolocating/posts_improved_geolocation.csv", min_count=20):
    """Plot results for the improved system with Hebrew support"""
    df = pd.read_csv(csv_path)

    locations_with_counts = df['location_name'].value_counts()
    filtered_locations = locations_with_counts[locations_with_counts >= min_count]

    if len(filtered_locations) == 0:
        print(f"No locations appear {min_count}+ times")
        return

    # Reshape Hebrew text for proper display
    def reshape_hebrew(text):
        return get_display(arabic_reshaper.reshape(text))

    filtered_locations.index = [reshape_hebrew(name) for name in filtered_locations.index]

    plt.figure(figsize=(12, 8))
    ax = filtered_locations.sort_values().plot(kind='barh')

    plt.title(reshape_hebrew(f'מיקומים עם {min_count}+ הזכרות - מערכת משופרת'), fontsize=14)
    plt.xlabel(reshape_hebrew('כמות הזכרות'), fontsize=12)
    plt.ylabel(reshape_hebrew('מיקום'), fontsize=12)

    plt.tight_layout()
    plt.show()

    # Show breakdown by type
    type_counts = df[df['location_name'].notna()]['location_type'].value_counts()
    print(f"\nBreakdown by location type:")
    for loc_type, count in type_counts.items():
        print(f"  {loc_type}: {count}")


def get_centroid_coordinates(geometry):
    """Get centroid coordinates, converting from Israel TM Grid if needed"""
    try:
        geom = shape(geometry)
        centroid = geom.centroid

        if abs(centroid.x) > 180 or abs(centroid.y) > 90:
            # Convert from Israel TM Grid to WGS84
            transformer = pyproj.Transformer.from_crs("EPSG:2039", "EPSG:4326", always_xy=True)
            return transformer.transform(centroid.x, centroid.y)
        return centroid.x, centroid.y
    except:
        return None, None


def create_location_lookup():
    """Create lookup dictionaries for neighborhoods and streets"""

    # Load GeoJSON files
    with open("../../input/jerusalem_neighborhoods.geojson", "r", encoding="utf-8") as f:
        neighborhoods = json.load(f)["features"]
    with open("../../input/jerusalem_streets.geojson", "r", encoding="utf-8") as f:
        streets = json.load(f)["features"]

    def build_lookup(features):
        lookup = {}
        for feature in features:
            props = feature["properties"]

            # Get OSM ID (like way/588415730 or node/278478009)
            osm_id = props.get("@id")
            if not osm_id:
                continue

            # Get Hebrew name
            name_he = None
            for name_field in ["name_he", "name_heb", "hebrew_name", "name"]:
                if name_field in props and props[name_field]:
                    name_he = props[name_field]
                    break

            if osm_id and name_he:
                lon, lat = get_centroid_coordinates(feature["geometry"])
                if lon is not None:
                    lookup[osm_id] = {"name": name_he, "longitude": lon, "latitude": lat}

        return lookup

    return build_lookup(neighborhoods), build_lookup(streets)


def add_coordinates_to_posts(df, output_csv_path="../../output/geolocating/posts_improved_geolocation.csv"):
    """Add longitude and latitude coordinates to geolocated posts"""

    df = df.copy()
    neighborhoods_lookup, streets_lookup = create_location_lookup()

    # Initialize coordinate columns
    df["longitude"] = np.nan
    df["latitude"] = np.nan
    df["coordinate_source"] = None

    # Add coordinates
    for idx, row in df.iterrows():
        if pd.isna(row.get("location_id")) or pd.isna(row.get("location_type")):
            continue

        location_id = row["location_id"]
        location_type = row["location_type"]

        lookup = neighborhoods_lookup if location_type == "neighborhood" else streets_lookup
        source = "neighborhood_centroid" if location_type == "neighborhood" else "street_centroid"

        if location_id in lookup:
            df.at[idx, "longitude"] = lookup[location_id]["longitude"]
            df.at[idx, "latitude"] = lookup[location_id]["latitude"]
            df.at[idx, "coordinate_source"] = source

    # Print stats and debug info
    coords_added = df["longitude"].notna().sum()
    total_located = df["location_name"].notna().sum()

    print(f"Added coordinates to {coords_added}/{total_located} located posts")
    print(f"Neighborhoods in lookup: {len(neighborhoods_lookup)}")
    print(f"Streets in lookup: {len(streets_lookup)}")

    # Show some example location_ids for debugging
    sample_posts = df[df["location_name"].notna()].head(3)
    print("Sample location_ids from posts:")
    for _, row in sample_posts.iterrows():
        print(f"  {row['location_type']}: {row['location_id']} -> {row['location_name']}")

    # Show some example keys from lookup
    if neighborhoods_lookup:
        print(f"Sample neighborhood keys: {list(neighborhoods_lookup.keys())[:3]}")
    if streets_lookup:
        print(f"Sample street keys: {list(streets_lookup.keys())[:3]}")

    if output_csv_path:
        df.to_csv(output_csv_path, encoding="utf-8-sig", index=False)
        print(f"Saved to: {output_csv_path}")

    return df

# File-based wrapper
def add_coordinates_to_posts_from_file(input_csv_path, output_csv_path="../../output/geolocating/posts_improved_geolocation.csv"):
    df = pd.read_csv(input_csv_path)
    if not output_csv_path:
        output_csv_path = input_csv_path.replace('.csv', '_with_coordinates.csv')
    return add_coordinates_to_posts(df, output_csv_path)


if __name__ == "__main__":
    # Configure matplotlib for Hebrew support
    import matplotlib

    matplotlib.rcParams['font.family'] = 'Arial'

    # Run the improved geolocation system
    df_results = geolocate_posts_improved()
    add_coordinates_to_posts(df_results)

    # Validate the coordinates
    add_coordinates_to_posts_from_file("../../output/geolocating/posts_improved_geolocation.csv", "../../output/geolocating/posts_improved_geolocation.csv")

    # Plot results
    #plot_improved_results()