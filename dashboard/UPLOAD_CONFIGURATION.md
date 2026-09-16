# Streamlit Upload Configuration

The project configures `server.maxUploadSize = 500` in `.streamlit/config.toml`.
This is a Streamlit browser-upload limit in megabytes, not a forecasting or model limitation.

Large production datasets should eventually use chunked or streaming ingestion rather than relying on browser upload.
