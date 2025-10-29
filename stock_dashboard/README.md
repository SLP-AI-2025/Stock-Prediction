## Frontend Research

Goal: Start a basic Streamlit app that loads our predictions.json and plots the prices

## How I ran this project:

python -m venv venv

source venv/bin/activate     # Mac/Linux  
venv\Scripts\activate        # Windows

pip install -r requirements.txt

streamlit run app.py

## Output

It loads predictions.json, lets the user pick a ticker, and plots that ticker’s closing prices over time

Below are screenshots of the 3 different graphs this dashboard shows:

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
  <img src="./example_output/AAPI.jpg" alt="AAPL chart" width="400" height="250" style="object-fit: cover;"/>
  <img src="./example_output/MSFT.jpg" alt="MSFT chart" width="400" height="250" style="object-fit: cover;"/>
  <img src="./example_output/SPY.jpg" alt="SPY chart" width="400" height="250" style="object-fit: cover;"/>
</p>

