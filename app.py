import streamlit as st
import requests
from dotenv import load_dotenv
import boto3
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
import json
from datetime import datetime, timedelta
from googleapiclient.discovery import build

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Constants
BASE_URL = 'http://api.weatherapi.com/v1/forecast.json'
EARTHQUAKE_URL = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
API_KEY = os.getenv('API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    if not docs:
        st.warning("No documents to index.")
        return
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

def get_mistral_llm():
    llm = Bedrock(model_id="mistral.mistral-large-2402-v1:0", client=bedrock,
                  model_kwargs={'max_tokens': 2000})
    return llm

def get_weather_forecast(location):
    params = {
        'key': API_KEY,
        'q': location,
        'days': 7
    }
    response = requests.get(BASE_URL, params=params)
    return response.json()

def get_earthquake_data(lat, lon):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30)
    
    params = {
        'format': 'geojson',
        'starttime': start_time.strftime('%Y-%m-%d'),
        'endtime': end_time.strftime('%Y-%m-%d'),
        'latitude': lat,
        'longitude': lon,
        'maxradiuskm': 300,
        'minmagnitude': 2.5
    }
    
    response = requests.get(EARTHQUAKE_URL, params=params)
    return response.json()

def generate_pdf(weather_data, earthquake_data, filename='data/weather_earthquake_data.pdf'):
    if not os.path.exists('data'):
        os.makedirs('data')

    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    width, height = letter

    def draw_text(text, x, y):
        can.drawString(x, y, text)
        return y - 15

    can.setFont("Helvetica-Bold", 16)
    y_position = height - 72
    y_position = draw_text(f"Weather and Earthquake Report for {weather_data['location']['name']}", 72, y_position)

    # Weather data
    can.setFont("Helvetica-Bold", 12)
    y_position = draw_text(f"Current Weather on {weather_data['current']['last_updated']}", 72, y_position)
    can.setFont("Helvetica", 12)
    current = weather_data['current']
    y_position = draw_text(f"Temperature: {current['temp_c']} °C", 72, y_position)
    y_position = draw_text(f"Condition: {current['condition']['text']}", 72, y_position)
    y_position = draw_text(f"Humidity: {current['humidity']} %", 72, y_position)
    y_position = draw_text(f"Wind: {current['wind_kph']} kph", 72, y_position)
    y_position = draw_text(f"Precipitation: {current['precip_mm']} mm", 72, y_position)

    can.setFont("Helvetica-Bold", 12)
    y_position = draw_text("7-Day Weather Forecast", 72, y_position)
    can.setFont("Helvetica", 12)
    forecast = weather_data['forecast']['forecastday']
    
    for day in forecast[:3]:  # Show only 3 days to save space
        date = day['date']
        day_info = day['day']
        y_position = draw_text(f"{date}: Max {day_info['maxtemp_c']}°C, Min {day_info['mintemp_c']}°C, {day_info['condition']['text']}", 72, y_position)

    # Earthquake data
    can.setFont("Helvetica-Bold", 12)
    y_position = draw_text("Recent Earthquake Activity", 72, y_position)
    can.setFont("Helvetica", 12)
    
    earthquakes = earthquake_data['features'][:5]  # Show only top 5 earthquakes
    for quake in earthquakes:
        props = quake['properties']
        y_position = draw_text(f"Magnitude {props['mag']}, {props['place']}, {datetime.fromtimestamp(props['time']/1000).strftime('%Y-%m-%d')}", 72, y_position)

    can.save()
    packet.seek(0)
    new_pdf = PdfReader(packet)

    if os.path.exists(filename):
        existing_pdf = PdfReader(open(filename, "rb"))
        output = PdfWriter()
        for i in range(len(existing_pdf.pages)):
            output.add_page(existing_pdf.pages[i])
        output.add_page(new_pdf.pages[0])
        output_stream = open(filename, "wb")
        output.write(output_stream)
        output_stream.close()
    else:
        with open(filename, "wb") as output_stream:
            output_stream.write(packet.getvalue())

def plot_weather_trends(weather_data):
    dates = [day['date'] for day in weather_data['forecast']['forecastday']]
    max_temps = [day['day']['maxtemp_c'] for day in weather_data['forecast']['forecastday']]
    min_temps = [day['day']['mintemp_c'] for day in weather_data['forecast']['forecastday']]
    precipitation = [day['day']['totalprecip_mm'] for day in weather_data['forecast']['forecastday']]

    df = pd.DataFrame({
        'Date': dates,
        'Max Temperature (°C)': max_temps,
        'Min Temperature (°C)': min_temps,
        'Precipitation (mm)': precipitation
    })

    fig = px.line(df, x='Date', y=['Max Temperature (°C)', 'Min Temperature (°C)', 'Precipitation (mm)'],
                  title=f"7-Day Weather Forecast for {weather_data['location']['name']}",
                  labels={'value': 'Value', 'variable': 'Metric'})
    
    return fig

def get_location_coordinates(location):
    geolocator = Nominatim(user_agent="weather_app")
    location_data = geolocator.geocode(location)
    if location_data:
        return location_data.latitude, location_data.longitude
    return None, None

def create_weather_earthquake_map(weather_data, earthquake_data):
    lat, lon = get_location_coordinates(weather_data['location']['name'])
    if lat and lon:
        m = folium.Map(location=[lat, lon], zoom_start=8)
        
        # Add weather marker
        folium.Marker(
            [lat, lon],
            popup=f"Current Temperature: {weather_data['current']['temp_c']}°C\n"
                  f"Condition: {weather_data['current']['condition']['text']}",
            tooltip=weather_data['location']['name'],
            icon=folium.Icon(color='blue', icon='cloud')
        ).add_to(m)
        
        # Add earthquake markers
        for quake in earthquake_data['features']:
            eq_lat, eq_lon = quake['geometry']['coordinates'][1], quake['geometry']['coordinates'][0]
            magnitude = quake['properties']['mag']
            place = quake['properties']['place']
            folium.CircleMarker(
                [eq_lat, eq_lon],
                radius=magnitude * 2,
                popup=f"Magnitude: {magnitude}<br>Location: {place}",
                color='red',
                fill=True,
                fillColor='red'
            ).add_to(m)
        
        return m
    return None

def generate_emergency_response_plan(weather_data, earthquake_data, llm):
    current_conditions = f"Temperature {weather_data['current']['temp_c']}°C, {weather_data['current']['condition']['text']}"
    
    forecast_summary = []
    for day in weather_data['forecast']['forecastday'][:3]:
        forecast_summary.append(f"{day['date']}: Max {day['day']['maxtemp_c']}°C, Min {day['day']['mintemp_c']}°C, {day['day']['condition']['text']}")
    
    forecast_str = "\n".join(forecast_summary)

    recent_earthquakes = [
        f"Magnitude {eq['properties']['mag']}, {eq['properties']['place']}, {datetime.fromtimestamp(eq['properties']['time']/1000).strftime('%Y-%m-%d')}"
        for eq in earthquake_data['features'][:5]
    ]
    earthquake_str = "\n".join(recent_earthquakes)

    prompt = f"""
    Based on the following weather forecast and recent earthquake activity for {weather_data['location']['name']}, generate a comprehensive emergency response plan:

    Current conditions: {current_conditions}

    3-day forecast:
    {forecast_str}

    Recent earthquakes:
    {earthquake_str}

    Please provide:
    1. Immediate actions for residents
    2. Key weather and seismic risks and potential emergencies
    3. Detailed preparedness measures for residents
    4. Specific recommendations for local emergency services
    5. A communication strategy for public safety announcements
    6. Long-term mitigation strategies

    Format the plan with clear headings and bullet points for easy readability. Prioritize the most critical aspects based on the severity of the weather and earthquake data.
    """

    response = llm(prompt)
    return response

def calculate_risk_score(weather_data, earthquake_data):
    risk_score = 0
    current = weather_data['current']
    forecast = weather_data['forecast']['forecastday']
    
    # Weather risks
    if current['temp_c'] > 35 or current['temp_c'] < 0:
        risk_score += 2
    
    if current['wind_kph'] > 60:
        risk_score += 2
    
    if current['precip_mm'] > 50:
        risk_score += 2
    
    if current['vis_km'] < 2:
        risk_score += 1
    
    # Check for extreme weather in the forecast
    for day in forecast:
        if day['day']['maxtemp_c'] > 40 or day['day']['mintemp_c'] < -10:
            risk_score += 1
        if day['day']['maxwind_kph'] > 80:
            risk_score += 1
        if day['day']['totalprecip_mm'] > 100:
            risk_score += 1
    
    # Earthquake risks
    for quake in earthquake_data['features']:
        magnitude = quake['properties']['mag']
        days_ago = (datetime.utcnow() - datetime.fromtimestamp(quake['properties']['time']/1000)).days
        if magnitude >= 6:
            risk_score += 3 if days_ago <= 7 else 2
        elif magnitude >= 5:
            risk_score += 2 if days_ago <= 7 else 1
        elif magnitude >= 4:
            risk_score += 1 if days_ago <= 7 else 0.5
    
    return min(risk_score, 10)  # Cap the risk score at 10

def get_ngo_contacts(location):
    # Use a geocoding service to get the coordinates of the location
    geolocator = Nominatim(user_agent="weather_app")
    location_data = geolocator.geocode(location)
    
    if location_data:
        lat, lon = location_data.latitude, location_data.longitude
        
        # Use these coordinates to query a hypothetical NGO database or API
        # For this example, we'll return mock data based on the coordinates
        ngos = [
            {"name": "Local Red Cross", "phone": f"+1-800-RED-{int(lat*lon)%10000:04d}"},
            {"name": "City Emergency Services", "phone": f"+1-911-{int(lat+lon)%10000:04d}"},
            {"name": "Regional Disaster Relief", "phone": f"+1-800-HELP-{int(lat-lon)%10000:04d}"},
        ]
        return ngos
    else:
        return []

def plot_disaster_analysis(weather_data, earthquake_data):
    # Weather risks
    dates = [day['date'] for day in weather_data['forecast']['forecastday']]
    max_temps = [day['day']['maxtemp_c'] for day in weather_data['forecast']['forecastday']]
    wind_speeds = [day['day']['maxwind_kph'] for day in weather_data['forecast']['forecastday']]
    precip = [day['day']['totalprecip_mm'] for day in weather_data['forecast']['forecastday']]

    # Earthquake data
    eq_dates = [datetime.fromtimestamp(eq['properties']['time']/1000).strftime('%Y-%m-%d') for eq in earthquake_data['features']]
    magnitudes = [eq['properties']['mag'] for eq in earthquake_data['features']]

    fig = go.Figure()

    # Weather data
    fig.add_trace(go.Scatter(x=dates, y=max_temps, name="Max Temperature (°C)"))
    fig.add_trace(go.Scatter(x=dates, y=wind_speeds, name="Max Wind Speed (kph)"))
    fig.add_trace(go.Bar(x=dates, y=precip, name="Precipitation (mm)"))

    # Earthquake data
    fig.add_trace(go.Scatter(x=eq_dates, y=magnitudes, mode='markers', 
                             name="Earthquakes", marker=dict(size=magnitudes*3, color='red')))

    fig.update_layout(title="Weather and Earthquake Analysis",
                      xaxis_title="Date",
                      yaxis_title="Value",
                      legend_title="Metrics",
                      hovermode="x unified")

    return fig

def plot_weather_patterns(weather_data):
    dates = [day['date'] for day in weather_data['forecast']['forecastday']]
    max_temps = [day['day']['maxtemp_c'] for day in weather_data['forecast']['forecastday']]
    min_temps = [day['day']['mintemp_c'] for day in weather_data['forecast']['forecastday']]
    precipitation = [day['day']['totalprecip_mm'] for day in weather_data['forecast']['forecastday']]
    wind_speed = [day['day']['maxwind_kph'] for day in weather_data['forecast']['forecastday']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=max_temps, name="Max Temperature"))
    fig.add_trace(go.Scatter(x=dates, y=min_temps, name="Min Temperature"))
    fig.add_trace(go.Bar(x=dates, y=precipitation, name="Precipitation"))
    fig.add_trace(go.Scatter(x=dates, y=wind_speed, name="Max Wind Speed", yaxis="y2"))

    fig.update_layout(
        title=f"Weather Patterns for {weather_data['location']['name']}",
        yaxis=dict(title="Temperature (°C) / Precipitation (mm)"),
        yaxis2=dict(title="Wind Speed (kph)", overlaying="y", side="right"),
        hovermode="x unified"
    )

    return fig

def analyze_disaster_risks(weather_data, earthquake_data, llm):
    location = weather_data['location']['name']
    current_temp = weather_data['current']['temp_c']
    current_condition = weather_data['current']['condition']['text']
    
    recent_earthquakes = [
        f"Magnitude {eq['properties']['mag']}, {eq['properties']['place']}"
        for eq in earthquake_data['features'][:3]
    ]
    earthquake_str = "\n".join(recent_earthquakes)
    
    prompt = f"""
    Analyze potential disaster risks for {location} based on the current weather, 7-day forecast, and recent earthquake activity:
    Current temperature: {current_temp}°C
    Current condition: {current_condition}

    Recent earthquakes:
    {earthquake_str}

    Provide:
    1. A brief summary of potential disasters that could affect {location}
    2. Detailed analysis of each potential disaster, including:
       - Likelihood of occurrence
       - Potential impact on the city and its residents
       - Factors contributing to the risk
    3. Recommendations for disaster preparedness and mitigation

    Consider both weather-related and seismic risks in your analysis.
    Present the information in a clear, detailed manner with bullet points for easy readability.
    """

    response = llm(prompt)
    return response

def get_weather_news_summary(location, weather_data, earthquake_data, llm):
    current_date = datetime.now().strftime("%Y-%m-%d")
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    query = f"{location} weather OR earthquake news after:{one_week_ago}"
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    
    res = service.cse().list(
        q=query,
        cx=GOOGLE_CSE_ID,
        num=5,
        sort="date"
    ).execute()
    
    news_items = []
    for item in res.get('items', []):
        news_items.append({
            'title': item['title'],
            'snippet': item['snippet'],
        })
    
    recent_earthquakes = [
        f"Magnitude {eq['properties']['mag']}, {eq['properties']['place']}"
        for eq in earthquake_data['features'][:3]
    ]
    earthquake_str = "\n".join(recent_earthquakes)
    
    news_summary_prompt = f"""
    Summarize the following weather and earthquake-related news for {location} in 3-4 lines each:

    {json.dumps(news_items, indent=2)}

    Current weather: {weather_data['current']['condition']['text']}, {weather_data['current']['temp_c']}°C

    Recent earthquakes:
    {earthquake_str}

    Provide a concise summary for each news item, focusing on its relevance to the current weather conditions, recent seismic activity, and any potential impacts on the city. Only include summaries for news items that are directly related to weather or earthquake events in {location}. If a news item is not relevant, do not include it in the summary.
    """

    summaries = llm(news_summary_prompt)
    return summaries

def main():
    st.set_page_config(page_title="ResqAI - Your Emergency Companion", page_icon="🌦️", layout="wide")
    
    st.title('🌦️ ResqAI - Your Emergency Companion')
    
    st.sidebar.header("Application Features")
    app_mode = st.sidebar.radio("Choose the app mode",
        ["Weather and Earthquake Data", "Disaster Analysis", "Emergency Response Plan", "Weather and Earthquake News"])
    
    location = st.text_input('Enter a location:', placeholder="e.g., New York, London, Tokyo")
    
    if st.button("Update Data", key="update_data"):
        if location:
            with st.spinner("Fetching and processing weather and earthquake data..."):
                weather_data = get_weather_forecast(location)
                lat, lon = get_location_coordinates(location)
                earthquake_data = get_earthquake_data(lat, lon)
                generate_pdf(weather_data, earthquake_data)
                docs = data_ingestion()
                vectorstore_faiss = get_vector_store(docs)
                st.session_state['weather_data'] = weather_data
                st.session_state['earthquake_data'] = earthquake_data
                st.session_state['vectorstore_faiss'] = vectorstore_faiss
                st.success("Weather and earthquake data updated and vector store refreshed!")
        else:
            st.warning("Please enter a location before updating data.")

    if app_mode == "Weather and Earthquake Data":
        if location and 'weather_data' in st.session_state and 'earthquake_data' in st.session_state:
            weather_data = st.session_state['weather_data']
            earthquake_data = st.session_state['earthquake_data']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Current Weather in {weather_data['location']['name']}")
                st.metric("Temperature", f"{weather_data['current']['temp_c']}°C")
                st.metric("Wind", f"{weather_data['current']['wind_kph']} kph, {weather_data['current']['wind_dir']}")
                st.metric("Condition", weather_data['current']['condition']['text'])
            
            with col2:
                st.subheader("7-Day Weather Forecast")
                fig = plot_weather_trends(weather_data)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Weather and Earthquake Map")
            m = create_weather_earthquake_map(weather_data, earthquake_data)
            if m:
                folium_static(m, width=1000, height=600)
            else:
                st.write("Unable to generate map for this location.")

            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Risk Assessment")
                risk_score = calculate_risk_score(weather_data, earthquake_data)
                st.metric("Current Risk Score", f"{risk_score:.1f}/10")
                
                if risk_score > 7:
                    st.warning("High risk detected. Here are some NGO contacts for emergency assistance:")
                    ngos = get_ngo_contacts(location)
                    for ngo in ngos:
                        st.write(f"🚨 {ngo['name']}: {ngo['phone']}")
            
            with col4:
                st.subheader("Weather Patterns")
                pattern_fig = plot_weather_patterns(weather_data)
                st.plotly_chart(pattern_fig, use_container_width=True)
            
            st.subheader("Recent Earthquakes")
            for quake in earthquake_data['features'][:5]:
                st.info(f"Magnitude {quake['properties']['mag']}, {quake['properties']['place']}, {datetime.fromtimestamp(quake['properties']['time']/1000).strftime('%Y-%m-%d')}")
        else:
            st.warning("Please enter a location and update data first.")

    elif app_mode == "Disaster Analysis":
        if location and 'weather_data' in st.session_state and 'earthquake_data' in st.session_state:
            weather_data = st.session_state['weather_data']
            earthquake_data = st.session_state['earthquake_data']
            llm = get_mistral_llm()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Disaster Risk Analysis for {weather_data['location']['name']}")
                analysis = analyze_disaster_risks(weather_data, earthquake_data, llm)
                st.markdown(analysis)
            
            with col2:
                st.subheader("Disaster Risk Visualization")
                disaster_fig = plot_disaster_analysis(weather_data, earthquake_data)
                st.plotly_chart(disaster_fig, use_container_width=True)
        else:
            st.warning("Please enter a location and update data first.")

    elif app_mode == "Emergency Response Plan":
        if location and 'weather_data' in st.session_state and 'earthquake_data' in st.session_state:
            weather_data = st.session_state['weather_data']
            earthquake_data = st.session_state['earthquake_data']
            llm = get_mistral_llm()
            plan = generate_emergency_response_plan(weather_data, earthquake_data, llm)
            st.subheader(f"Emergency Response Plan for {weather_data['location']['name']}")
            st.markdown(plan)
            
            risk_score = calculate_risk_score(weather_data, earthquake_data)
            if risk_score > 7:
                st.warning("High risk detected. Here are some NGO contacts for emergency assistance:")
                ngos = get_ngo_contacts(location)
                for ngo in ngos:
                    st.info(f"🚨 {ngo['name']}: {ngo['phone']}")
            
            st.download_button(
                label="Download Emergency Response Plan",
                data=plan,
                file_name="emergency_response_plan.txt",
                mime="text/plain"
            )
        else:
            st.warning("Please enter a location and update data first.")

    elif app_mode == "Weather and Earthquake News":
        if location and 'weather_data' in st.session_state and 'earthquake_data' in st.session_state:
            weather_data = st.session_state['weather_data']
            earthquake_data = st.session_state['earthquake_data']
            llm = get_mistral_llm()
            news_summaries = get_weather_news_summary(location, weather_data, earthquake_data, llm)
            st.subheader(f"Weather and Earthquake News Summaries for {location}")
            st.markdown(news_summaries)
        else:
            st.warning("Please enter a location and update data first.")

    st.sidebar.info("This application was developed for the Disaster Management Hackathon. It aims to provide comprehensive weather and earthquake data analysis to help communities prepare for and respond to natural disasters.")
    st.sidebar.success("For emergencies, always contact your local authorities first.")

    st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        Made with love by Team Tetris ❤️
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == '__main__':
    main()