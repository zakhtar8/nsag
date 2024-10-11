import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium import FeatureGroup
from shapely.geometry import Point, MultiPolygon, shape
import alphashape
from streamlit_folium import st_folium
import ee
import math

# Initialize Earth Engine
service_account = 'general@ee-zainabakh1998.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(
    service_account, 'ee-zainabakh1998-099a685cfd97.json')
ee.Initialize(credentials)


st.set_page_config(
	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
	page_title=None,  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)

# Streamlit application
st.title("Children in Non-State Active Areas")

# Initialize session state variables if not already initialized
if 'df_high_activity' not in st.session_state:
    st.session_state.df_high_activity = {}
if 'country_population_data' not in st.session_state:
    st.session_state.country_population_data = {}
if 'inter1_polygons' not in st.session_state:
    st.session_state.inter1_polygons = {}
if 'population_data' not in st.session_state:
    st.session_state.population_data = {}
if 'plot_concave_hulls_flag' not in st.session_state:
    st.session_state.plot_concave_hulls_flag = False
if 'retrieve_populations_flag' not in st.session_state:
    st.session_state.retrieve_populations_flag = False

# Option to select an existing file or upload a CSV file
use_existing_file = st.sidebar.checkbox(
    "Use 2023 All Countries.csv", key='use_existing_file')

if use_existing_file:
    # Load the existing file from the same directory
    df = pd.read_csv('./2023 all countries.csv')
    st.write("Using existing 2023 countries.csv file:")
    st.write(df.head())
else:
    # Upload CSV file for processing
    uploaded_file = st.sidebar.file_uploader(
        "Upload your ACLED Actors (CSV) file", type=["csv"], key='uploaded_file')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(df.head())
    else:
        df = None  # Set to None if no file is uploaded or selected

if df is not None:
    # Country dictionary mapping
    country_dict = {
        'NGA': 'Nigeria',
        'SDN': 'Sudan',
        'MLI': 'Mali',
        'MMR': 'Myanmar',
        'UKR': 'Ukraine',
        'BFA': 'Burkina Faso',
        'NER': 'Niger',
        'COD': 'Democratic Republic of Congo',
        'COL': 'Colombia',
        'HTI': 'Haiti'
    }

    # Multiselect for selecting multiple countries
    selected_countries = st.sidebar.multiselect(
        "Select countries", options=list(country_dict.values()), key='selected_countries')

    # Map selected countries to their codes
    selected_country_codes = [code for code,
                              name in country_dict.items() if name in selected_countries]
    if selected_country_codes:
        st.write(f"Selected country codes: {', '.join(selected_country_codes)}")
    else:
        st.write("No countries selected.")

    # Check if any countries are selected
    if selected_countries:
        # Input for alpha value with a slider (now in the sidebar)
        alpha_value = st.sidebar.slider("Alpha Value for Concave Hulls",
                                        min_value=0.1, max_value=5.0,
                                        value=1.25,  # Default value, no need for session state
                                        step=0.05, key='alpha_value_slider')

        # The slider automatically updates st.session_state.alpha_value

        # Country inter1 color mapping (kept for internal use)
        inter1_options = {
            2: 'Blue (inter1=2)',
            3: 'Green (inter1=3)',
            4: 'Red (inter1=4)',
            8: 'Purple (inter1=8)'
        }


        # Function to calculate median event counts
        def calculate_median_event_counts(df, country):
            inter1_values = [2, 3, 4, 8]  # Hardcoded inter1 values
            st.write(
                f"Filtering data for country: {country} and inter1 values: {inter1_values}...")

            # Filter the dataframe based on the country and inter1 values
            df_filtered = df[(df['country'].str.contains(country, case=False, na=False)) &
                             (df['inter1'].isin(inter1_values))]

            # Check if the filtered dataframe has any data
            if df_filtered.empty:
                st.error(
                    f"No data found for country '{country}' with inter1 values {inter1_values}.")
                return None

            # Group by 'admin2', 'location', 'inter1', 'longitude', and 'latitude'
            grouped_data = df_filtered.groupby(['admin2', 'location', 'inter1', 'longitude',
                                               'latitude']).size().reset_index(name='count')

            # Get the median count for each group by 'admin2' and 'inter1'
            event_count_median = grouped_data.groupby(
                ['admin2', 'inter1'])['count'].transform('median')

            # Add the median column back to the grouped data
            grouped_data['median'] = event_count_median

            # Classify groups with higher activity levels
            grouped_data['activity_level'] = (
                grouped_data['count'] >= grouped_data['median']).astype(int)

            # Filter the rows where the activity level is 1 (high activity)
            df_high_activity = grouped_data[grouped_data['activity_level'] == 1]

            return df_high_activity

        # Function to calculate population for the entire country and area in km²
        def calculate_population_by_group(countrycode):
            st.write(f"Calculating population for country code {countrycode}...")

            # Fetch WorldPop data for 2020
            worldpop_agesex = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex") \
                .filterDate('2020') \
                .filterMetadata('country', 'equals', countrycode) \
                .first()

            # Load the country AOI from the asset
            admin0_country = ee.FeatureCollection(
                'projects/ee-zainabakh1998/assets/mapbox_admin0')

            # Filter AOI by country code
            aoi_feature = admin0_country.filter(
                ee.Filter.eq('ISO_A3', countrycode)).first()
            aoi = aoi_feature.geometry()

            # Convert AOI to GeoJSON Feature
            aoi_geojson = aoi.getInfo()

            if aoi_geojson is None:
                st.error("Failed to retrieve AOI geometry from Earth Engine.")
                return None

            try:
                aoi_shape = shape(aoi_geojson)
            except Exception as e:
                st.error(f"Error converting AOI GeoJSON to shapely geometry: {e}")
                return None

            # Create GeoDataFrame from shapely geometry
            aoi_gdf = gpd.GeoDataFrame({'geometry': [aoi_shape]}, crs='EPSG:4326')  # Assuming WGS84

            # Project to a projection suitable for area calculation (e.g., EPSG:3857)
            aoi_gdf_projected = aoi_gdf.to_crs(epsg=3857)
            area_m2 = aoi_gdf_projected.geometry.area.iloc[0]
            area_km2 = area_m2 / 1e6  # Convert to km²

            # Clip the population data to the AOI
            worldpop_agesex_clipped = worldpop_agesex.clip(aoi)

            # Perform population statistics reduction
            stats2 = worldpop_agesex_clipped.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=100,  # The dataset's resolution is 100m
                maxPixels=1e10,
                bestEffort=True
            )

            # Extract population data and round down to nearest integer
            population_keys = ['population', 'F_0', 'F_1', 'F_5',
                               'F_10', 'M_0', 'M_1', 'M_5', 'M_10']
            data = {}
            for key in population_keys:
                value = stats2.get(key).getInfo()
                data[key] = int(value) if value is not None else 0

            # Add area to the data
            data['area_km2'] = round(area_km2, 2)  # Rounded to 2 decimal places

            return data

        # Function to generate concave hulls and display both points and polygons on the map
        def calculate_and_plot_concave_hulls(df, alpha):
            inter1_values = df['inter1'].unique()
            m = folium.Map(zoom_start=2)

            # Colors for the different inter1 categories
            inter1_colors = {
                2: 'blue',
                3: 'green',
                4: 'red',
                8: 'purple'
            }

            all_bounds = []  # For fitting the map bounds

            inter1_polygons = {}  # Dictionary to store polygons for each inter1

            for inter1 in inter1_values:
                filtered_df = df[df['inter1'] == inter1]
                geometry = [Point(xy) for xy in zip(
                    filtered_df['longitude'], filtered_df['latitude'])]
                gdf = gpd.GeoDataFrame(filtered_df, geometry=geometry)
                coords = [(point.x, point.y) for point in gdf.geometry]

                # Generate the concave hull
                if len(coords) >= 3:
                    alpha_shape = alphashape.alphashape(coords, alpha=alpha)
                    inter1_polygons[inter1] = alpha_shape  # Store the polygon

                    if isinstance(alpha_shape, MultiPolygon):
                        geojson_data = gpd.GeoSeries(
                            [alpha_shape]).to_json()
                    else:
                        geojson_data = gpd.GeoSeries(
                            [alpha_shape]).to_json()

                    # Create a FeatureGroup for each inter1 to contain both polygon and points
                    fg = FeatureGroup(name=f"inter1={inter1}")

                    # Add polygons (concave hulls) to the FeatureGroup
                    folium.GeoJson(
                        geojson_data,
                        name=f"inter1={inter1} Polygon",
                        style_function=lambda x, color=inter1_colors.get(inter1, 'gray'): {
                            'fillColor': color,
                            'color': color,
                            'weight': 2,
                            'fillOpacity': 0.25
                        }
                    ).add_to(fg)
                else:
                    # Not enough points to create a polygon
                    fg = FeatureGroup(name=f"inter1={inter1} (No polygon)")

                # Add points to the FeatureGroup
                for _, row in filtered_df.iterrows():
                    lat, lon = row['latitude'], row['longitude']
                    popup_content = f"inter1: {row['inter1']}<br>Location: {row['location']}<br>Count: {row['count']}"
                    folium.CircleMarker(
                        location=(lat, lon),
                        radius=2,
                        color=inter1_colors.get(inter1, 'black'),
                        fill=True,
                        fill_color=inter1_colors.get(inter1, 'black'),
                        fill_opacity=0.6,
                        popup=folium.Popup(popup_content, max_width=200)
                    ).add_to(fg)
                    all_bounds.append([lat, lon])

                # Add the FeatureGroup to the map
                fg.add_to(m)

            # Fit map to bounds
            if all_bounds:
                min_lat = min(b[0] for b in all_bounds)
                min_lon = min(b[1] for b in all_bounds)
                max_lat = max(b[0] for b in all_bounds)
                max_lon = max(b[1] for b in all_bounds)
                m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

            # Add LayerControl to toggle inter1 layers (polygons and points together)
            folium.LayerControl().add_to(m)

            return m, inter1_polygons

        # Function to calculate population within a given polygon and area in km²
        def calculate_population_in_polygon(polygon_shape, countrycode):
            # Convert the shapely polygon to GeoJSON
            geojson = gpd.GeoSeries(
                [polygon_shape]).__geo_interface__['features'][0]['geometry']

            # Create an Earth Engine geometry from the GeoJSON
            ee_geometry = ee.Geometry(geojson)

            # Fetch WorldPop data for 2020
            worldpop_agesex = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex") \
                .filterDate('2020') \
                .filterMetadata('country', 'equals', countrycode) \
                .first()

            # Clip the population data to the polygon area
            worldpop_agesex_clipped = worldpop_agesex.clip(ee_geometry)

            # Perform population statistics reduction
            stats2 = worldpop_agesex_clipped.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=ee_geometry,
                scale=100,  # The dataset's resolution is 100m
                maxPixels=1e10,
                bestEffort=True
            )

            # Extract population data and round down to nearest integer
            population_keys = ['population', 'F_0', 'F_1', 'F_5',
                               'F_10', 'M_0', 'M_1', 'M_5', 'M_10']
            data = {}
            for key in population_keys:
                value = stats2.get(key).getInfo()
                data[key] = int(value) if value is not None else 0

            # Calculate area in km²
            polygon_gdf = gpd.GeoDataFrame({'geometry': [polygon_shape]}, crs='EPSG:4326')  # Assuming WGS84
            # Project to a projection suitable for area calculation (e.g., EPSG:3857)
            polygon_gdf_projected = polygon_gdf.to_crs(epsg=3857)
            area_m2 = polygon_gdf_projected.geometry.area.iloc[0]
            area_km2 = area_m2 / 1e6  # Convert to km²
            data['area_km2'] = round(area_km2, 2)  # Rounded to 2 decimal places

            return data

        # Buttons for each step
        st.header("Processing Steps")

        # Button to calculate median event counts
        if st.button("1. Calculate Median Event Counts", key='calc_median'):
            for country_name in selected_countries:
                country_code = [
                    code for code, name in country_dict.items() if name == country_name][0]
                if country_code not in st.session_state.df_high_activity:
                    df_high_activity = calculate_median_event_counts(
                        df, country_name)
                    if df_high_activity is not None:
                        st.session_state.df_high_activity[country_code] = df_high_activity
                    else:
                        st.session_state.df_high_activity[country_code] = None

        # Button to calculate country population
        if st.button("2. Calculate Country-Wide Total Population and Children Population", key='calc_country_pop'):
            for country_code in selected_country_codes:
                if country_code not in st.session_state.country_population_data:
                    country_pop_data = calculate_population_by_group(
                        country_code)
                    if country_pop_data:
                        st.session_state.country_population_data[country_code] = country_pop_data
                    else:
                        st.session_state.country_population_data[country_code] = None

        # Button to plot concave hulls
        if st.button("3. Plot Concave Hulls for the 4 Actor Types", key='plot_concave_hulls_button'):
            st.session_state.plot_concave_hulls_flag = True

        # Button to retrieve population in inter1 polygons
        if st.button("4. Retrieve Total Population and Children Population in the 4 Concave Polygons", key='retrieve_pop_button'):
            st.session_state.retrieve_populations_flag = True

        # Create tabs for each selected country
        country_tabs = st.tabs(selected_countries)

        # Display outputs in tabs without re-processing data
        for idx, country_name in enumerate(selected_countries):
            # Get the country code
            country_code = [
                code for code, name in country_dict.items() if name == country_name][0]

            with country_tabs[idx]:
                st.header(f"{country_name} ({country_code})")

                # Display high activity data
                df_high_activity = st.session_state.df_high_activity.get(
                    country_code)
                if df_high_activity is not None:
                    st.write("High activity data based on median counts:")
                    # Ensure 'count' and 'median' are integers
                    df_display = df_high_activity.copy()
                    population_columns = ['count', 'median']
                    for col in population_columns:
                        df_display[col] = df_display[col].astype(int)
                    st.write(df_display)
                else:
                    st.write("No high activity data. Please click '1. Calculate Median Event Counts' to generate the table.")

                # Display country population data
                country_pop_data = st.session_state.country_population_data.get(
                    country_code)
                if country_pop_data is not None:
                    st.write("Country Population Data:")
                    # Convert to DataFrame
                    df_country_population = pd.DataFrame(
                        [country_pop_data])
                    st.write(df_country_population)
                else:
                    st.write("No country population data. Please click '2. Calculate Country-Wide Total Population and Children Population' to generate the table.")

                # Generate and display the concave map if data is available
                if st.session_state.plot_concave_hulls_flag:
                    df_high_activity = st.session_state.df_high_activity.get(country_code)
                    if df_high_activity is not None:
                        # Use the alpha value from the slider
                        concave_map, inter1_polygons = calculate_and_plot_concave_hulls(
                            df_high_activity,
                            alpha=alpha_value  # Use the value from the slider directly
                        )
                        st.session_state.inter1_polygons[country_code] = inter1_polygons

                        st.write("Concave Hull Map (You can adjust the alpha slider to see the dynamic updates):")
                        # Dynamically render the map when alpha value changes
                        concave_map, inter1_polygons = calculate_and_plot_concave_hulls(
                            df_high_activity, alpha=alpha_value  # Directly use the slider's value
                        )
                        st_folium(concave_map, width=700, height=500, key=f'concave_map_{country_code}_alpha_{alpha_value}')

                    else:
                        st.write("No map available. Please click '1. Calculate Median Event Counts' to generate the map.")
                else:
                    st.write("No map available. Please click '3. Plot Concave Hulls for the 4 Actor Types' to generate the map.")

                # Retrieve and display population data within each inter1 polygon
                if st.session_state.retrieve_populations_flag:
                    if country_code in st.session_state.inter1_polygons and st.session_state.inter1_polygons[country_code] is not None:
                        inter1_polygons = st.session_state.inter1_polygons[country_code]
                        population_data = []
                        for inter1, polygon_shape in inter1_polygons.items():
                            if polygon_shape is not None:
                                pop_data = calculate_population_in_polygon(
                                    polygon_shape, country_code)
                                pop_data['inter1'] = inter1_options.get(
                                    inter1, f"inter1={inter1}")
                                population_data.append(pop_data)
                        st.session_state.population_data[country_code] = population_data
                    else:
                        st.session_state.population_data[country_code] = None

                # Display population data within each inter1 polygon
                population_data = st.session_state.population_data.get(
                    country_code)
                if population_data is not None and len(population_data) > 0:
                    st.write("Population Data within Each Concave Polygon:")
                    df_population = pd.DataFrame(population_data)
                    st.write(df_population)
                else:
                    st.write("No population data in polygons. Please click '4. Retrieve Total Population and Children Population in the 4 Concave Polygons' to calculate.")

    else:
        st.warning("Please select at least one country.")

    # Footer
    st.write("Developed by Zainab Akhtar.")

else:
    st.warning("Please upload the required CSV file or select to use the existing one.")
