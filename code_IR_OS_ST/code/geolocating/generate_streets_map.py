import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd
from shapely.geometry import Point


def create_heatmap_data(posts_df, streets_gdf, neighborhoods_gdf, name_col, column_name, filter_func=None):
    """
    Generic function to create heatmap data based on any column.
    Returns list of [lat, lon, weight] for HeatMap with weighted values.
    """
    heatmap_points = []

    # Helper function to get location point (same as in main function)
    def get_location_point(geom):
        if geom.geom_type == 'Point':
            return (geom.x, geom.y)
        elif geom.geom_type == 'LineString':
            coords = list(geom.coords)
            return coords[len(coords) // 2]
        else:
            centroid = geom.centroid
            return (centroid.x, centroid.y)

    # Apply custom filter or default filter
    if filter_func:
        filtered_posts = filter_func(posts_df)
    else:
        filtered_posts = posts_df[
            posts_df[column_name].notna() &
            posts_df['location_id'].notna() &
            posts_df[name_col].notna()
            ]

    print(f"Found {len(filtered_posts)} posts with {column_name} data")

    # Process both street and neighborhood posts
    for location_type in ['street', 'neighborhood']:
        location_posts = filtered_posts[filtered_posts['location_type'] == location_type]
        gdf = streets_gdf if location_type == 'street' else neighborhoods_gdf

        for _, post in location_posts.iterrows():
            try:
                for loc_id in str(post['location_id']).split(';'):
                    loc_id = loc_id.strip()
                    match = gdf[gdf['id'].astype(str) == loc_id]
                    if not match.empty:
                        geom = match.iloc[0].geometry
                        pt = get_location_point(geom)
                        weight = float(post[column_name]) if pd.notna(post[column_name]) else 1.0
                        heatmap_points.append([pt[1], pt[0], weight])
            except Exception as e:
                print(f"Error processing {location_type} {column_name} post: {e}")

    print(f"Created {len(heatmap_points)} {column_name} heatmap points")
    return heatmap_points


def create_posts_map(posts_csv_path="../../output/geolocating/geolocated_posts.csv",
                     streets_geojson_path="../../input/jerusalem_streets.geojson",
                     neighborhoods_geojson_path="../../input/jerusalem_neighborhoods.geojson",
                     municipality_geojson_path="../../input/jerusalem_Municipality.geojson",
                     arab_neighborhoods_geojson_path="../../input/jerusalem_arab_neighborhoods.geojson",
                     haredi_neighborhoods_geojson_path="../../input/jerusalem_haredi_neighborhoods.geojson",
                     gov_neighborhoods_geojson_path="../../input/jerusalem_government_neighborhoods.geojson",
                     output_html_path="../../output/geolocating/posts_map.html"):
    """
    Create an interactive map showing Facebook posts with their matched streets and neighborhoods,
    with Jerusalem municipality boundary and special neighborhood classifications.
    """

    print("Loading data...")
    posts_df = pd.read_csv(posts_csv_path)
    streets_gdf = gpd.read_file(streets_geojson_path)
    neighborhoods_gdf = gpd.read_file(neighborhoods_geojson_path)
    municipality_gdf = gpd.read_file(municipality_geojson_path)
    arab_neighborhoods_gdf = gpd.read_file(arab_neighborhoods_geojson_path)
    haredi_neighborhoods_gdf = gpd.read_file(haredi_neighborhoods_geojson_path)
    gov_neighborhoods_gdf = gpd.read_file(gov_neighborhoods_geojson_path)
    # Add 'id' column if missing
    if 'id' not in streets_gdf.columns:
        streets_gdf['id'] = streets_gdf.index
    if 'id' not in neighborhoods_gdf.columns:
        neighborhoods_gdf['id'] = neighborhoods_gdf.index

    # Find location name column
    name_col = next(
        (col for col in ['street_name', 'location_name', 'matched_name', 'name'] if col in posts_df.columns), None)
    if not name_col:
        raise ValueError("❌ None of the expected location name columns were found in the posts CSV.")
    print(f"✅ Using '{name_col}' as the location name column.")

    # Filter posts by location type
    posts_with_streets = posts_df[
        (posts_df['location_type'] == 'street') &
        posts_df['location_id'].notna() &
        posts_df[name_col].notna()
        ]
    posts_with_neighborhoods = posts_df[
        (posts_df['location_type'] == 'neighborhood') &
        posts_df['location_id'].notna() &
        posts_df[name_col].notna()
        ]

    # Create base map centered on Jerusalem
    m = folium.Map(location=[31.7683, 35.2137], zoom_start=12, tiles='OpenStreetMap')
    folium.TileLayer('cartodbpositron', name='Light Map').add_to(m)

    def get_location_point(geom):
        """Extract a representative point from any geometry type"""
        if geom.geom_type == 'Point':
            return (geom.x, geom.y)
        elif geom.geom_type == 'LineString':
            coords = list(geom.coords)
            return coords[len(coords) // 2]
        else:
            centroid = geom.centroid
            return (centroid.x, centroid.y)

    # Collect points for heatmap and markers
    heatmap_points = []
    post_points = []
    matched_street_ids = set()
    matched_neighborhood_ids = set()

    # Process street posts
    for _, post in posts_with_streets.iterrows():
        try:
            for loc_id in str(post['location_id']).split(';'):
                loc_id = loc_id.strip()
                match = streets_gdf[streets_gdf['id'].astype(str) == loc_id]
                if not match.empty:
                    geom = match.iloc[0].geometry
                    matched_street_ids.add(loc_id)
                    pt = get_location_point(geom)

                    # Add to heatmap (lat, lon format)
                    heatmap_points.append([pt[1], pt[0]])

                    # Add to post points with metadata
                    post_points.append({
                        'lat': pt[1],
                        'lon': pt[0],
                        'location_name': post[name_col],
                        'location_type': 'street',
                        'post_content': str(post.get('content', 'N/A'))[:100] + '...' if len(
                            str(post.get('content', 'N/A'))) > 100 else str(post.get('content', 'N/A')),
                        'post_date': post.get('date', 'N/A')
                    })
        except Exception as e:
            print(f"Error processing street post: {e}")

    # Process neighborhood posts
    for _, post in posts_with_neighborhoods.iterrows():
        try:
            for loc_id in str(post['location_id']).split(';'):
                loc_id = loc_id.strip()
                match = neighborhoods_gdf[neighborhoods_gdf['id'].astype(str) == loc_id]
                if not match.empty:
                    geom = match.iloc[0].geometry
                    matched_neighborhood_ids.add(loc_id)
                    pt = get_location_point(geom)

                    # Add to heatmap (lat, lon format)
                    heatmap_points.append([pt[1], pt[0]])

                    # Add to post points with metadata
                    post_points.append({
                        'lat': pt[1],
                        'lon': pt[0],
                        'location_name': post[name_col],
                        'location_type': 'neighborhood',
                        'post_content': str(post.get('content', 'N/A'))[:100] + '...' if len(
                            str(post.get('content', 'N/A'))) > 100 else str(post.get('content', 'N/A')),
                        'post_date': post.get('date', 'N/A')
                    })
        except Exception as e:
            print(f"Error processing neighborhood post: {e}")

    print(f"Found {len(heatmap_points)} post locations")
    print(f"Matched {len(matched_street_ids)} streets and {len(matched_neighborhood_ids)} neighborhoods")

    # Add Jerusalem municipality boundary
    municipality_layer = folium.FeatureGroup(name='Jerusalem Municipality', show=True)
    try:
        for _, row in municipality_gdf.iterrows():
            if row.geometry and not row.geometry.is_empty:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'none',
                        'color': 'black',
                        'weight': 3,
                        'fillOpacity': 0,
                        'opacity': 0.8,
                        'dashArray': '5, 5'
                    },
                    popup=False,
                    tooltip=False
                ).add_to(municipality_layer)
        municipality_layer.add_to(m)
        print("✅ Added Jerusalem municipality boundary")
    except Exception as e:
        print(f"❌ Error adding municipality boundary: {e}")

    # Add individual post markers with clustering
    marker_cluster = MarkerCluster(name='Individual Posts', show=False)
    for point in post_points:
        color = 'blue' if point['location_type'] == 'street' else 'orange'
        icon_symbol = 'road' if point['location_type'] == 'street' else 'home'

        popup_html = f"""
        <div style="width: 200px;">
            <b>Location:</b> {point['location_name']}<br>
            <b>Type:</b> {point['location_type']}<br>
            <b>Date:</b> {point['post_date']}<br>
            <b>Content:</b> {point['post_content']}
        </div>
        """

        folium.Marker(
            location=[point['lat'], point['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{point['location_name']} ({point['location_type']})",
            icon=folium.Icon(color=color, icon=icon_symbol, prefix='fa')
        ).add_to(marker_cluster)
    marker_cluster.add_to(m)

    # Add Arab neighborhoods layer as polygons
    arab_layer = folium.FeatureGroup(name='Arab Neighborhoods', show=False)
    try:
        for _, row in arab_neighborhoods_gdf.iterrows():
            if row.geometry and not row.geometry.is_empty:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'red',
                        'color': 'darkred',
                        'weight': 2,
                        'fillOpacity': 0.4,
                        'opacity': 0.8
                    },
                    popup=folium.Popup(f"Arab Neighborhood: {row.get('SCHN_NAME', 'Unknown')}", max_width=200),
                    tooltip=f"Arab: {row.get('SCHN_NAME', 'Unknown')}"
                ).add_to(arab_layer)
        arab_layer.add_to(m)
        print("✅ Added Arab neighborhoods layer")
    except Exception as e:
        print(f"❌ Error adding Arab neighborhoods: {e}")

    # Add Haredi neighborhoods layer as polygons
    haredi_layer = folium.FeatureGroup(name='Haredi Neighborhoods', show=False)
    try:
        for _, row in haredi_neighborhoods_gdf.iterrows():
            if row.geometry and not row.geometry.is_empty:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'purple',
                        'color': 'darkviolet',
                        'weight': 2,
                        'fillOpacity': 0.4,
                        'opacity': 0.8
                    },
                    popup=folium.Popup(f"Haredi Neighborhood: {row.get('SCHN_NAME', 'Unknown')}", max_width=200),
                    tooltip=f"Haredi: {row.get('SCHN_NAME', 'Unknown')}"
                ).add_to(haredi_layer)
        haredi_layer.add_to(m)
        print("✅ Added Haredi neighborhoods layer")
    except Exception as e:
        print(f"❌ Error adding Haredi neighborhoods: {e}")
    # Add gov layer
    government_layer = folium.FeatureGroup(name='Government Neighborhoods', show=False)
    try:
        for _, row in gov_neighborhoods_gdf.iterrows():
            if row.geometry and not row.geometry.is_empty:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'gray',
                        'color': 'gray',
                        'weight': 2,
                        'fillOpacity': 0.4,
                        'opacity': 0.8
                    },
                    popup=folium.Popup(f"Gov Neighborhood: {row.get('SCHN_NAME', 'Unknown')}", max_width=200),
                    tooltip=f"Gov: {row.get('SCHN_NAME', 'Unknown')}"
                ).add_to(government_layer)
        government_layer.add_to(m)
        print("✅ Added Government neighborhoods layer")
    except Exception as e:
        print(f"❌ Error adding Government neighborhoods: {e}")

    # Add regular post density heatmap
    if heatmap_points:
        HeatMap(heatmap_points, name='Post Density', radius=10, blur=15, max_zoom=14).add_to(m)

    # Add multiple heatmap layers
    heatmap_configs = [
        {
            'column': 'rent_price',
            'name': 'Rent Price Heatmap',
            'filter_func': lambda df: df[(df['rent_price'].notna()) & (df['rent_price'] > 0) &
                                         (df['location_id'].notna()) & (df[name_col].notna())],
            'gradient': {0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        },
        {
            'column': 'num_rooms',
            'name': 'Number of Rooms Heatmap',
            'filter_func': lambda df: df[(df['num_rooms'].notna()) & (df['num_rooms'] > 0) &
                                         (df['location_id'].notna()) & (df[name_col].notna())],
            'gradient': {0.2: 'lightblue', 0.4: 'blue', 0.6: 'purple', 0.8: 'darkviolet', 1.0: 'indigo'}
        },
        {
            'column': 'is_sublet',
            'name': 'Sublet Properties Heatmap',
            'filter_func': lambda df: df[(df['is_sublet'].notna()) & (df['is_sublet'] == True) &
                                         (df['location_id'].notna()) & (df[name_col].notna())],
            'gradient': {0.4: 'orange', 0.7: 'darkorange', 1.0: 'red'}
        },
        {
            'column': 'is_shabbat_kosher',
            'name': 'Kosher Properties Heatmap',
            'filter_func': lambda df: df[(df['is_shabbat_kosher'].notna()) & (df['is_shabbat_kosher'] == True) &
                                         (df['location_id'].notna()) & (df[name_col].notna())],
            'gradient': {0.4: 'lightgreen', 0.7: 'green', 1.0: 'darkgreen'}
        }
    ]

    for config in heatmap_configs:
        try:
            heatmap_points = create_heatmap_data(
                posts_df, streets_gdf, neighborhoods_gdf, name_col,
                config['column'], config['filter_func']
            )
            if heatmap_points:
                HeatMap(
                    heatmap_points,
                    name=config['name'],
                    radius=15,
                    blur=20,
                    max_zoom=14,
                    show=False,
                    gradient=config['gradient']
                ).add_to(m)
                print(f"✅ Added {config['name'].lower()}")
            else:
                print(f"⚠️ No {config['column']} data found for heatmap")
        except Exception as e:
            print(f"❌ Error adding {config['name'].lower()}: {e}")

    # Add legend (updated to include all heatmaps)
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 320px; height: 300px; 
                background-color: white; border: 2px solid grey; z-index: 9999; 
                font-size: 12px; padding: 15px; line-height: 1.4;">
        <h4 style="margin: 0 0 10px 0; font-size: 16px;">Map Legend</h4>
        <p style="margin: 3px 0; display: flex; align-items: center;">
            <span style="background: rgba(0,0,255,0.3); width: 14px; height: 14px; display: inline-block; border-radius: 7px; margin-right: 8px;"></span>
            Post Density Heatmap
        </p>
        <p style="margin: 3px 0; display: flex; align-items: center;">
            <span style="background: linear-gradient(to right, blue, cyan, lime, yellow, red); width: 14px; height: 14px; display: inline-block; border-radius: 7px; margin-right: 8px;"></span>
            Rent Price Heatmap
        </p>
        <p style="margin: 3px 0; display: flex; align-items: center;">
            <span style="background: linear-gradient(to right, lightblue, blue, purple, darkviolet, indigo); width: 14px; height: 14px; display: inline-block; border-radius: 7px; margin-right: 8px;"></span>
            Number of Rooms Heatmap
        </p>
        <p style="margin: 3px 0; display: flex; align-items: center;">
            <span style="background: linear-gradient(to right, orange, darkorange, red); width: 14px; height: 14px; display: inline-block; border-radius: 7px; margin-right: 8px;"></span>
            Sublet Properties Heatmap
        </p>
        <p style="margin: 3px 0; display: flex; align-items: center;">
            <span style="background: linear-gradient(to right, lightgreen, green, darkgreen); width: 14px; height: 14px; display: inline-block; border-radius: 7px; margin-right: 8px;"></span>
            Kosher Properties Heatmap
        </p>
        <p style="margin: 3px 0; display: flex; align-items: center;">
            <span style="width: 14px; height: 14px; background-color: red; border: 1px solid darkred; margin-right: 8px; display: inline-block;"></span>
            Arab Neighborhoods
        </p>
        <p style="margin: 3px 0; display: flex; align-items: center;">
            <span style="width: 14px; height: 14px; background-color: purple; border: 1px solid darkviolet; margin-right: 8px; display: inline-block;"></span>
            Haredi Neighborhoods
        </p>
         <p style="margin: 3px 0; display: flex; align-items: center;">
            <span style="width: 14px; height: 14px; background-color: gray; border: 1px solid darkviolet; margin-right: 8px; display: inline-block;"></span>
            Government/Industrial/University Areas
        </p>
        <p style="margin: 3px 0; display: flex; align-items: center;">
            <span style="width: 14px; height: 3px; background-color: black; border-top: 2px dashed black; margin-right: 8px; display: inline-block;"></span>
            Municipality Boundary
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add layer control
    folium.LayerControl().add_to(m)

    # Fit map to municipality bounds if available
    try:
        if not municipality_gdf.empty:
            bounds = municipality_gdf.total_bounds  # [minx, miny, maxx, maxy]
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        else:
            # Fallback to neighborhoods bounds
            bounds = neighborhoods_gdf.total_bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except Exception as e:
        print(f"Error fitting map bounds: {e}")

    # Save the map
    print(f"Saving map to {output_html_path}")
    m.save(output_html_path)
    print("✅ Map created successfully!")
    return m


def main():
    try:
        map_obj = create_posts_map()
        if map_obj:
            print("Map generated successfully.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()