import streamlit as st
import requests

# Use your OpenWeatherMap API key
API_KEY = "15b87e2a5c283e035e598a00cd37ab3a"

# Function to get weather data
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_info = {
            'City': data['name'],
            'Temperature (°C)': data['main']['temp'],
            'Humidity (%)': data['main']['humidity'],
            'Wind Speed (m/s)': data['wind']['speed'],
            'Description': data['weather'][0]['description']
        }
        return weather_info
    else:
        return {"Error": f"Failed to retrieve data: {response.status_code}"}

# Streamlit app layout
st.title("Weather Forecast App")
st.write("Enter a city name to get the current weather information:")

# Add custom CSS for background color
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #1E90FF; /* Blue background */
    }
    .stButton>button {
        background-color: #ffffff; /* White button */
        color: #1E90FF; /* Blue text for button */
        border: none;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input for city
city = st.text_input("City Name")

if st.button("Get Weather"):
    if city:
        weather_info = get_weather(city)
        if "Error" in weather_info:
            st.error(weather_info["Error"])
        else:
            st.success(f"Weather in {weather_info['City']}:")
            st.write(f"**Temperature:** {weather_info['Temperature (°C)']}°C")
            st.write(f"**Humidity:** {weather_info['Humidity (%)']}%")
            st.write(f"**Wind Speed:** {weather_info['Wind Speed (m/s)']} m/s")
            st.write(f"**Description:** {weather_info['Description']}")
    else:
        st.warning("Please enter a city name.")
