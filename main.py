from flask import Flask, request, render_template
import requests

app = Flask(__name__)

# Use your OpenWeatherMap API key
API_KEY = "15b87e2a5c283e035e598a00cd37ab3a"

@app.route('/', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':
        city = request.form['city']
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description']
            }
            return render_template('weather.html', weather_info=weather_info)
        else:
            error_message = f"Failed to retrieve data: {response.status_code}"
            return render_template('weather.html', error_message=error_message)

    return render_template('weather.html')

if __name__ == '__main__':
    app.run(debug=True)
