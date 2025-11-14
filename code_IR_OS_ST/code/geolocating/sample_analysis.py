import pandas as pd
import numpy as np
import json
from pathlib import Path


def load_geojson_data():
    """Load Jerusalem streets and neighborhoods GeoJSON data"""
    streets_path = "../../input/jerusalem_streets.geojson"
    neighborhoods_path = "../../input/jerusalem_neighborhoods.geojson"

    streets_data = {}
    neighborhoods_data = {}

    try:
        with open(streets_path, 'r', encoding='utf-8') as f:
            streets_geojson = json.load(f)
            for feature in streets_geojson.get('features', []):
                props = feature.get('properties', {})
                name = props.get('name') or props.get('street_name') or props.get('STREET_NAME')
                if name:
                    streets_data[name.lower()] = {
                        'type': props.get('highway') or props.get('type') or 'street',
                        'name': name,
                        'properties': props
                    }
    except FileNotFoundError:
        print(f"Warning: Streets file not found at {streets_path}")
    except Exception as e:
        print(f"Warning: Error loading streets data: {e}")

    try:
        with open(neighborhoods_path, 'r', encoding='utf-8') as f:
            neighborhoods_geojson = json.load(f)
            for feature in neighborhoods_geojson.get('features', []):
                props = feature.get('properties', {})
                name = props.get('name') or props.get('SHEM_SHCHU') or props.get('neighborhood')
                if name:
                    neighborhoods_data[name.lower()] = {
                        'type': 'neighborhood',
                        'name': name,
                        'properties': props
                    }
    except FileNotFoundError:
        print(f"Warning: Neighborhoods file not found at {neighborhoods_path}")
    except Exception as e:
        print(f"Warning: Error loading neighborhoods data: {e}")

    return streets_data, neighborhoods_data


def get_feature_info(location_name, streets_data, neighborhoods_data):
    """Get feature type and properties from GeoJSON data"""
    if pd.isna(location_name):
        return 'unknown', None

    location_lower = str(location_name).lower().strip()

    # Check exact match first
    if location_lower in streets_data:
        return streets_data[location_lower]['type'], streets_data[location_lower]

    if location_lower in neighborhoods_data:
        return neighborhoods_data[location_lower]['type'], neighborhoods_data[location_lower]

    # Check partial matches (for cases where location_name might have additional info)
    for street_name, street_info in streets_data.items():
        if street_name in location_lower or location_lower in street_name:
            return street_info['type'], street_info

    for neighborhood_name, neighborhood_info in neighborhoods_data.items():
        if neighborhood_name in location_lower or location_lower in neighborhood_name:
            return neighborhood_info['type'], neighborhood_info

    return 'unmatched', None


def analyze_validation_sample(csv_path="../../output/geolocating/validation_sample_improved.csv"):
    """
    Analyze validation sample results with concise output for review
    Returns: Summary statistics and breakdown by method/confidence/feature_type/mismatch_length
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return "Validation file not found. Please create validation sample first."

    # Load GeoJSON data
    print("Loading Jerusalem streets and neighborhoods data...")
    streets_data, neighborhoods_data = load_geojson_data()
    print(f"Loaded {len(streets_data)} streets and {len(neighborhoods_data)} neighborhoods")

    # Check if validation is complete
    if df['is_correct'].isna().all() or df['is_correct'].eq('').all():
        return "Validation not completed. Please fill 'is_correct' column with True/False values."

    # Convert is_correct to boolean, handling various input formats including NA
    df['is_correct_bool'] = df['is_correct'].astype(str).str.lower().map({
        'true': True, 't': True, '1': True, 'yes': True, 'y': True,
        'false': False, 'f': False, '0': False, 'no': False, 'n': False,
        'nan': None, 'na': None, '': None
    })

    # Get feature information from GeoJSON data
    feature_info = df['location_name'].apply(lambda x: get_feature_info(x, streets_data, neighborhoods_data))
    df['feature_type'] = feature_info.apply(lambda x: x[0])
    df['feature_data'] = feature_info.apply(lambda x: x[1])

    # Calculate mismatch length for incorrect predictions
    def calculate_mismatch_length(row):
        if pd.isna(row['is_correct_bool']) or row['is_correct_bool']:
            return None  # Only calculate for incorrect predictions

        location_name = str(row['location_name']) if pd.notna(row['location_name']) else ""
        notes = str(row['notes']) if pd.notna(row['notes']) else ""

        # Simple length difference as a proxy for mismatch severity
        # You might want to implement more sophisticated mismatch calculation here
        return len(location_name)

    df['mismatch_length'] = df.apply(calculate_mismatch_length, axis=1)

    # Split into categories
    validated_df = df[df['is_correct_bool'].notna()].copy()  # True/False only
    unvalidated_count = df['is_correct_bool'].isna().sum()  # NA count

    if len(validated_df) == 0:
        return f"No validation completed yet. {len(df)} entries waiting for validation (all NA)."

    # Overall accuracy
    total_validated = len(validated_df)
    correct_predictions = validated_df['is_correct_bool'].sum()
    overall_accuracy = correct_predictions / total_validated * 100

    print("=== VALIDATION ANALYSIS ===")
    print(f"Total samples: {len(df)}")
    print(f"Validated: {len(validated_df)} | Pending: {unvalidated_count}")
    print(f"Correct: {correct_predictions} | Incorrect: {len(validated_df) - correct_predictions}")
    print(f"Overall accuracy: {overall_accuracy:.1f}%\n")

    # Accuracy by method
    print("ACCURACY BY METHOD:")
    method_stats = validated_df.groupby('location_method').agg({
        'is_correct_bool': ['count', 'sum', 'mean']
    }).round(3)
    method_stats.columns = ['Total', 'Correct', 'Accuracy']
    method_stats['Accuracy_pct'] = (method_stats['Accuracy'] * 100).round(1)

    for method in method_stats.index:
        stats = method_stats.loc[method]
        print(f"  {method}: {stats['Correct']}/{stats['Total']} ({stats['Accuracy_pct']}%)")

    # Accuracy by type
    print("\nACCURACY BY TYPE:")
    type_stats = validated_df.groupby('location_type').agg({
        'is_correct_bool': ['count', 'sum', 'mean']
    }).round(3)
    type_stats.columns = ['Total', 'Correct', 'Accuracy']
    type_stats['Accuracy_pct'] = (type_stats['Accuracy'] * 100).round(1)

    for loc_type in type_stats.index:
        stats = type_stats.loc[loc_type]
        print(f"  {loc_type}: {stats['Correct']}/{stats['Total']} ({stats['Accuracy_pct']}%)")

    # NEW: Accuracy by feature type (from GeoJSON data)
    print("\nACCURACY BY FEATURE TYPE:")
    feature_type_stats = validated_df.groupby('feature_type').agg({
        'is_correct_bool': ['count', 'sum', 'mean']
    }).round(3)
    feature_type_stats.columns = ['Total', 'Correct', 'Accuracy']
    feature_type_stats['Accuracy_pct'] = (feature_type_stats['Accuracy'] * 100).round(1)

    for feature_type in feature_type_stats.index:
        stats = feature_type_stats.loc[feature_type]
        print(f"  {feature_type}: {stats['Correct']}/{stats['Total']} ({stats['Accuracy_pct']}%)")

    # Confidence analysis
    print("\nCONFIDENCE ANALYSIS:")
    correct_confidence = validated_df[validated_df['is_correct_bool'] == True]['location_confidence']
    incorrect_confidence = validated_df[validated_df['is_correct_bool'] == False]['location_confidence']

    print(
        f"Correct predictions - confidence: mean={correct_confidence.mean():.1f}, median={correct_confidence.median():.1f}")
    print(
        f"Incorrect predictions - confidence: mean={incorrect_confidence.mean():.1f}, median={incorrect_confidence.median():.1f}")

    # Confidence ranges
    print("\nACCURACY BY CONFIDENCE RANGE:")
    validated_df['conf_range'] = pd.cut(validated_df['location_confidence'],
                                        bins=[0, 80, 90, 95, 100],
                                        labels=['70-80', '80-90', '90-95', '95-100'],
                                        include_lowest=True)

    conf_stats = validated_df.groupby('conf_range').agg({
        'is_correct_bool': ['count', 'sum', 'mean']
    }).round(3)
    conf_stats.columns = ['Total', 'Correct', 'Accuracy']
    conf_stats['Accuracy_pct'] = (conf_stats['Accuracy'] * 100).round(1)

    for conf_range in conf_stats.index:
        if pd.notna(conf_range):
            stats = conf_stats.loc[conf_range]
            print(f"  {conf_range}: {stats['Correct']}/{stats['Total']} ({stats['Accuracy_pct']}%)")

    # NEW: Mismatch length analysis for incorrect predictions
    incorrect_df = validated_df[validated_df['is_correct_bool'] == False]

    if len(incorrect_df) > 0:
        print(f"\nMISMATCH LENGTH ANALYSIS ({len(incorrect_df)} incorrect predictions):")
        mismatch_lengths = incorrect_df['mismatch_length'].dropna()

        if len(mismatch_lengths) > 0:
            print(
                f"Location name length - mean: {mismatch_lengths.mean():.1f}, median: {mismatch_lengths.median():.1f}")
            print(f"Range: {mismatch_lengths.min():.0f} - {mismatch_lengths.max():.0f} characters")

            # Categorize by length ranges
            length_ranges = pd.cut(mismatch_lengths,
                                   bins=[0, 10, 20, 50, 100, float('inf')],
                                   labels=['Very Short (≤10)', 'Short (11-20)', 'Medium (21-50)', 'Long (51-100)',
                                           'Very Long (>100)'],
                                   include_lowest=True)

            length_counts = length_ranges.value_counts().sort_index()
            print("Distribution by length:")
            for length_range, count in length_counts.items():
                print(f"  {length_range}: {count} ({count / len(mismatch_lengths) * 100:.1f}%)")

    # Problem areas (incorrect predictions)
    print(f"\nPROBLEM AREAS ({len(validated_df) - correct_predictions} incorrect):")

    if len(incorrect_df) > 0:
        # Group by method and type for incorrect predictions
        problem_method = incorrect_df['location_method'].value_counts()
        problem_type = incorrect_df['location_type'].value_counts()
        problem_feature_type = incorrect_df['feature_type'].value_counts()

        print("By method:")
        for method, count in problem_method.items():
            print(f"  {method}: {count}")

        print("By type:")
        for loc_type, count in problem_type.items():
            print(f"  {loc_type}: {count}")

        print("By feature type:")
        for feature_type, count in problem_feature_type.items():
            print(f"  {feature_type}: {count}")

        # Show a few examples of incorrect predictions with enhanced info
        print(f"\nSample incorrect predictions:")
        sample_incorrect = incorrect_df.head(3)[
            ['location_name', 'location_confidence', 'location_method', 'feature_type', 'mismatch_length', 'notes']]
        for idx, row in sample_incorrect.iterrows():
            note = row['notes'] if pd.notna(row['notes']) and row['notes'] else "No notes"
            mismatch_info = f", length: {row['mismatch_length']:.0f}" if pd.notna(row['mismatch_length']) else ""
            print(
                f"  - {row['location_name']} ({row['location_confidence']:.0f}%, {row['location_method']}, {row['feature_type']}{mismatch_info}) - {note}")

    # Summary recommendations
    print(f"\n=== SUMMARY ===")
    if overall_accuracy >= 85:
        print("✓ Good overall accuracy")
    elif overall_accuracy >= 70:
        print("⚠ Moderate accuracy - room for improvement")
    else:
        print("✗ Low accuracy - needs significant improvement")

    # Feature type insights
    if len(validated_df) > 0:
        best_feature_type = feature_type_stats['Accuracy_pct'].idxmax()
        worst_feature_type = feature_type_stats['Accuracy_pct'].idxmin()
        print(
            f"Best performing feature type: {best_feature_type} ({feature_type_stats.loc[best_feature_type, 'Accuracy_pct']:.1f}%)")
        print(
            f"Worst performing feature type: {worst_feature_type} ({feature_type_stats.loc[worst_feature_type, 'Accuracy_pct']:.1f}%)")

    return {
        'overall_accuracy': overall_accuracy,
        'total_validated': total_validated,
        'pending_validation': unvalidated_count,
        'method_accuracy': method_stats['Accuracy_pct'].to_dict(),
        'type_accuracy': type_stats['Accuracy_pct'].to_dict(),
        'feature_type_accuracy': feature_type_stats['Accuracy_pct'].to_dict(),
        'mismatch_length_stats': {
            'mean': mismatch_lengths.mean() if len(incorrect_df) > 0 and len(mismatch_lengths) > 0 else None,
            'median': mismatch_lengths.median() if len(incorrect_df) > 0 and len(mismatch_lengths) > 0 else None
        },
        'geojson_match_stats': {
            'streets_loaded': len(streets_data),
            'neighborhoods_loaded': len(neighborhoods_data),
            'matched_features': len(validated_df[validated_df['feature_type'] != 'unmatched']),
            'unmatched_features': len(validated_df[validated_df['feature_type'] == 'unmatched'])
        }
    }


def create_validation_sample_improved(csv_path="../../output/geolocating/posts_improved_geolocation.csv",
                                      sample_size=100):
    """Create validation sample for the improved system"""
    df = pd.read_csv(csv_path)

    df_with_locations = df[df['location_name'].notna()].copy()
    print(f"Posts with locations: {len(df_with_locations)}")

    if len(df_with_locations) < sample_size:
        sample_df = df_with_locations.copy()
    else:
        sample_df = df_with_locations.sample(n=sample_size, random_state=22).copy()

    validation_df = pd.DataFrame({
        'post_content': sample_df['content'],
        'location_name': sample_df['location_name'],
        'location_confidence': sample_df['location_confidence'],
        'location_type': sample_df['location_type'],
        'location_method': sample_df['location_method'],
        'is_correct': '',  # To be filled manually (True/False)
        'notes': ''
    })

    output_path = "../../output/geolocating/validation_sample_improved.csv"
    validation_df.to_csv(output_path, encoding="utf-8-sig", index=False)

    print(f"Validation sample saved to: {output_path}")
    print(f"Sample size: {len(validation_df)}")
    print("\nTo validate: fill 'is_correct' column with True/False")

    return validation_df


def main():
    #create_validation_sample_improved()
    analyze_validation_sample()


if __name__ == "__main__":
    main()