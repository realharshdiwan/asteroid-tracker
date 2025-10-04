# Google Colab Setup for Asteroid Tracker
# Run this in a Colab notebook cell

!pip install streamlit pandas plotly numpy requests scipy matplotlib

# Create the app file
with open('atroapp.py', 'w') as f:
    # Copy your atroapp.py content here
    pass

# Run the app
!streamlit run atroapp.py --server.port 8501 --server.address 0.0.0.0
