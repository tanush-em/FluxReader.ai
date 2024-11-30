@echo off
REM Set up Python virtual environment and activate it
if not exist env-fluxreaderai (
    echo Creating virtual environment...
    python -m venv env-fluxreaderai
)

echo Activating virtual environment...
call env-fluxreaderai\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the Streamlit application
echo Starting the application...
streamlit run app.py

REM Wait for Streamlit to close
echo Press Ctrl+C to stop the application and close the terminal.
cmd /k "call env-fluxreaderai\Scripts\deactivate & exit"

