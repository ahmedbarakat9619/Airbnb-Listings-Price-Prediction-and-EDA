import joblib
import streamlit as st 
import pandas as pd 
import plotly.express as px 
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
from geopy.distance import great_circle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

st.set_page_config("Airbnb Listings Price Prediction - 10 European Cities")
model = joblib.load('XGB_model.pkl')
df3 = pd.read_csv('df3.csv')
df_metro_stations = pd.read_csv('metro_stations.csv')

def inputs():
    #Inputs
    room_type = st.selectbox("Specify the room type",("Private room", "Entire home/apt", "Shared room"),) #Room Type
    st.write("You selected:", room_type)
    
    person_capacity = st.select_slider("Specify maximum capacity per place", options=[2, 3, 4, 5, 6],) #person_capacity
    st.write("You selected:", person_capacity)
    
    is_superhost = st.radio("Is the host superhost??",[True, False],) #is_superhost
    if is_superhost == True:
        st.write("You selected: Superhost")
    else :
        st.write("You selected: Normal Host")
        
    cleanliness_rating = st.select_slider("How clean is the listing, 10 is the most clean", options=[2, 3, 4, 5, 6, 7, 8, 9, 10],) 
    st.write("You selected:", cleanliness_rating)
    
    guest_satisfaction_overall = st.slider("What is overall guest satisfaction rating of the listing?", 0, 100, 1)
    st.write("You selected:", guest_satisfaction_overall)
    
    bedrooms = st.select_slider("How many bedrooms?", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],) 
    st.write("You selected" ,bedrooms, "bedrooms")
    
    week_time = st.radio("Time of the reservation?",['weekdays', 'weekends'],)
    st.write("You selected:", week_time)
          
    
    def create_city_map(city_name, centre_coords, initial_coords):
        global listing_location, lat, lng, dist_centre
        st.write("**Click on the listing place on the map below**")
        m = folium.Map(location=initial_coords, zoom_start=12) #Create map of the city
        m.add_child(folium.LatLngPopup())  # Add a clickable popup to get coordinates
        output = st_folium(m, width=700, height=500)  # Display the map using Streamlit's st_folium

        # Check if the user has clicked on the map to get coordinates
        if output['last_clicked']:
            lat = output['last_clicked']['lat']
            lng = output['last_clicked']['lng']
            st.write("Latitude:", lat)
            st.write("Longitude:", lng)

            listing_location = (lat, lng)
            dist_centre = great_circle(listing_location, centre_coords).km       # Calculate the distance to the city center
            st.write("Distance to city center:", round(dist_centre, 2), "Km")
        else:
            st.write("Click on the map to get coordinates.")

    # Get city input from the user
    city = st.selectbox("Select a city:", ["amsterdam", "athens", "barcelona", "berlin", "budapest", "lisbon", "london", "paris", "rome", "vienna"])
    
    def distance_metro(city, df_metro_stations):
        global metro_dist
        df_metro = df_metro_stations[df_metro_stations['city'] == city]
        df_metro['distance'] = df_metro.apply(lambda row: great_circle(listing_location, (row['lat'], row['lon'])).km, axis=1)
        minimum_distance = df_metro['distance'].min()
        nearest_metro_lat = df_metro[df_metro['distance'] == minimum_distance]['lat']
        nearest_metro_lng = df_metro[df_metro['distance'] == minimum_distance]['lon']
        nearest_metro_name = df_metro[df_metro['distance'] == minimum_distance]['name']

        metro_dist = minimum_distance
        nearest_metro_name = nearest_metro_name[nearest_metro_name.index[0]]

        st.write(f"Distance to nearest metro station, {nearest_metro_name}, is: {round(metro_dist, 2)} Km")        

    # Define coordinates for each city's center and initial map view
    city_coords = {
        'amsterdam': {'centre': (52.3731, 4.8926), 'initial': (52.3676, 4.9041)},
        'athens': {'centre': (37.9755, 23.7348), 'initial': (37.9838, 23.7275)},
        'barcelona': {'centre': (41.3870, 2.1701), 'initial': (41.3851, 2.1734)},
        'berlin': {'centre': (52.5219, 13.4132), 'initial': (52.5200, 13.4050)},
        'budapest': {'centre': (47.4975, 19.0513), 'initial': (47.4979, 19.0402)},
        'lisbon': {'centre': (38.7139, -9.1395), 'initial': (38.7223, -9.1393)},
        'london': {'centre': (51.5080, -0.1281), 'initial': (51.5074, -0.1278)},
        'paris': {'centre': (48.8566, 2.3522), 'initial': (48.8566, 2.3522)},
        'rome': {'centre': (41.9028, 12.4963), 'initial': (41.9028, 12.4964)},
        'vienna': {'centre': (48.2082, 16.3738), 'initial': (48.2082, 16.3738)}
    }

    # Call the function to create a map based on the user's city selection
    if city in city_coords:
        create_city_map(city, city_coords[city]['centre'], city_coords[city]['initial'])
        distance_metro(city, df_metro_stations)
        
        
    #Input Data Preprocessing:
    
    if room_type == 'Private room':
        room_type = 2
    elif room_type == 'Entire home/apt':
        room_type = 3
    else: 
        room_type = 1
        
    if week_time == 'weekdays':
        week_time = 1
    elif week_time == 'weekends':
        week_time = 2

    city_dummy_mapping = {
    'amsterdam': 'city_amsterdam',
    'athens': 'city_athens',
    'barcelona': 'city_barcelona',
    'berlin': 'city_berlin',
    'budapest': 'city_budapest',
    'lisbon': 'city_lisbon',
    'london': 'city_london',
    'paris': 'city_paris',
    'rome': 'city_rome',
    'vienna': 'city_vienna'
    }

    dummy_vars = {
    'city_amsterdam': False,
    'city_athens': False,
    'city_barcelona': False,
    'city_berlin': False,
    'city_budapest': False,
    'city_lisbon': False,
    'city_london': False,
    'city_paris': False,
    'city_rome': False,
    'city_vienna': False
    }

    if city in city_dummy_mapping:
        dummy_vars[city_dummy_mapping[city]] = True

    
    input_df = pd.DataFrame([{'room_type': room_type,
                'person_capacity': person_capacity,
                'is_superhost': is_superhost,
                'cleanliness_rating': cleanliness_rating,
                'guest_satisfaction_overall': guest_satisfaction_overall,
                'bedrooms': bedrooms,
                'dist_centre': dist_centre,
                'metro_dist': metro_dist,
                'lng': lng,
                'lat': lat,
                'week_time': week_time,
                **dummy_vars}])
    
    if st.button('Predict Price of The Listing!'):
        predict(input_df)    
    
def predict(input_df):
    price = model.predict(input_df)
    rounded_price = round(float(price[0]), 2)
    st.subheader(f":red[Predicted price of the listing is {rounded_price} Euros]")
    

    
def about():
    st.title('Airbnb Listings Price Prediction') 
    st.header("About the project:", divider="gray")  
    st.markdown("**This project aims to predict the price of Airbnb listings across 10 big european cities, which are:**")
    st.markdown("1- Amsterdam")
    st.markdown("2- Barcelona")
    st.markdown("3- Rome")
    st.markdown("4- London")
    st.markdown("5- Paris")
    st.markdown("6- Lisbon")
    st.markdown("7- Budapest")
    st.markdown("8- Berlin")
    st.markdown("9- Athens")
    st.markdown("10- Vienna")
    
    st.subheader("Project Dataset:", divider="gray")
    st.markdown("Project is based on the following dataset from Kaggle:")
    st.link_button("Check the dataset", "https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities")
    st.markdown("The dataset contains over 50K rows and about 20 features including the target feature.")


    st.subheader("Project drive:", divider="gray")
    st.markdown("This is a final project for Certified data sceince diploma in Epsilon Ai institute, Egypt.")
    
    st.subheader("Application summary:", divider="gray")
    st.markdown("You can navigate from the left of the page in order to predict a listing price by choosing its location on the map of each city, and other relevant features that may affect the price.")
    st.markdown("Also, navigate to analysis and insights page in order to see some insights about the data.")
    
    st.subheader("About the author:", divider="gray")
    st.markdown("Name : Ahmed Mostafa Barakat")
    st.markdown("Email : ahmedbarakat9619@gmail.com")
    st.markdown("Phone : +201095517102")

    
    
def plots():
    st.header("Analysis and Insights: ", divider="gray")  
    
    st.subheader("Features explanation:", divider="gray")
    st.markdown("1. **price**: The total price of the listing")
    st.markdown("2. **room_type**: Type of room (private/shared/entire home/apt).")
    st.markdown("3. **is_superhost**: Boolean value indicating if the host is a superhost or not.")
    st.markdown("4. **guest_satisfaction_overall**: Overall rating from guests comparing all listings offered by the host")
    st.markdown("5. **bedrooms**: Number of bedrooms.")
    st.markdown("6. **dist_centre**: Distance from the city center.")
    st.markdown("9. **metro_dist**: Distance from the nearest metro.")
    st.markdown("10. **attr_index_norm**: Index demonstrating how near the listing is to city attractions, normalized from 0 to 100.")
    st.markdown("10. **rest_index_norm**: Index demonstrating how near the listing is to reataurants, normalized from 0 to 100.")
    st.markdown("11. **lng** & **lat** coordinates: For location identification.")
    st.markdown("12. **city**: City name.")
    st.markdown("13. **week_time**: Weekday or Weekend.")
    st.markdown("14. **price_cat_per_city**: Price category for each city (High, Average, Low), this an engineered features to show more insights about price categories.")
    st.markdown("15. **distance_category**: This is an engineered feature that combines the distance from centre and metro together.")

    st.subheader("Dataframe Head:", divider="gray")
    st.markdown("The following dataframe head is after feature engineering and outliers handling in the target column.")
    st.dataframe(df3.head())
    
    st.subheader("Plots and insights:", divider="gray")
    st.markdown("1- Price Distribution across the cities (red: mean, blue: median):")
    #Visualizing the price differnce between each city
    fig = plt.figure(figsize=(14, 8))
    sns.violinplot(x='city', y='price', data=df3, inner=None, color=".8")
    sns.boxplot(x='city', y='price', data=df3, showmeans=True, meanline=True, meanprops={"color": "red", "linewidth": 2},medianprops={"color": "blue", "linewidth": 2})
    plt.title('Price Distribution Across Cities', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.markdown("2- Distribution of Price on deifferent types of accomadation with variable person capacity:")
    fig = plt.figure(figsize=(14, 8))
    sns.scatterplot(data=df3, x=df3['room_type'], y=df3['price'], hue=df3['person_capacity'], palette='viridis')
    plt.title('Distribution of Price on deifferent types of accomadation with variable person capacity:')
    st.pyplot(fig)
    
    st.markdown("3- Cleanliness Vs Price")
    fig = plt.figure(figsize=(14, 8))
    sns.scatterplot(data=df3, x=df3['cleanliness_rating'], y=df3['price'], hue=df3['guest_satisfaction_overall'], palette='viridis')
    plt.title('Cleanliness Vs Price')
    st.pyplot(fig)
    
    st.markdown("4- Attractions vs Price")
    fig = plt.figure(figsize=(14, 8))
    sns.scatterplot(data=df3, x=df3['attr_index_norm'], y=df3['price'], palette='viridis')
    plt.title('Attractions vs Price')
    st.pyplot(fig)
    
    
    st.markdown("5- Centre Distance vs Price")
    fig = plt.figure(figsize=(14, 8))
    sns.scatterplot(data=df3, x=df3['dist_centre'], y=df3['price'])
    plt.title('Centre Distance vs Price')
    st.pyplot(fig)
    
    st.markdown("6- Metro Distance vs Price")
    fig = plt.figure(figsize=(14, 8))
    sns.scatterplot(data=df3, x=df3['metro_dist'], y=df3['price'], palette='viridis')
    plt.title('Metro Distance vs Price')
    st.pyplot(fig)
    
    st.markdown("7- Distance Category vs Price Category")
    fig = plt.figure(figsize=(14, 8))
    sns.countplot(data=df3, x=df3['price_cat_per_city'], hue=df3['distance_category'])
    plt.title('Distance Category vs Price Category')
    st.pyplot(fig)
    
    st.markdown("8- Week Time vs Price Category")
    fig = plt.figure(figsize=(14, 8))
    sns.countplot(data=df3, x=df3['price_cat_per_city'], hue=df3['week_time'], palette='viridis')
    plt.title('Week Time vs Price Category')
    st.pyplot(fig)
    
    st.markdown("9- Acoom. Type dist. over price categories")
    fig = plt.figure(figsize=(14, 8))
    sns.countplot(data=df3, x='price_cat_per_city', hue='room_type')
    plt.title('Acoom. Type dist. over price categories')
    st.pyplot(fig)
    
    st.markdown("10- Listings distribution and price categories over cities")
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    for i, (city, ax) in enumerate(zip(df3['city'].unique(), axes.flatten())):
        sns.scatterplot(data=df3[df3['city'] == city], x='lng', y='lat', hue='price_cat_per_city', ax=ax)
        ax.set_title(city)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("11- Listings attractions index and price categories over cities")
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    for i, (city, ax) in enumerate(zip(df3['city'].unique(), axes.flatten())):
        sns.scatterplot(data=df3[df3['city'] == city], x='lng', y='lat', hue='attr_index_norm', ax=ax)
        ax.set_title(city)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("12- Listings restaurants index and price categories over cities")
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    for i, (city, ax) in enumerate(zip(df3['city'].unique(), axes.flatten())):
        sns.scatterplot(data=df3[df3['city'] == city], x='lng', y='lat', hue='rest_index_norm', ax=ax)
        ax.set_title(city)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("13- Ditribution across cities showing price and distance categories")
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    for i, (city, ax) in enumerate(zip(df3['city'].unique(), axes.flatten())):
        sns.scatterplot(data=df3[df3['city'] == city], x='lng', y='lat', hue='distance_category', size = 'price_cat_per_city', size_order=['High','Average','Low'] , ax=ax)
        ax.set_title(city)
    plt.tight_layout()
    st.pyplot(fig)
  
    
page = st.sidebar.selectbox('Select Page',['About the Project','Predict','Analysis and Insights'])
    
if page == 'About the Project':
    about()
elif page == 'Predict':
    inputs()
else:
    plots()





