# This file is for Streamlit Cloud deployment
# It imports and runs the main application

from Video_monitoring_app import StreamlitApp

if __name__ == "__main__":
    app = StreamlitApp()
    app.run() 