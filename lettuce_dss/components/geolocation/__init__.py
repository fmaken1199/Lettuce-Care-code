from pathlib import Path

def location_button(key="farm_location"):
    from streamlit.components.v1 import declare_component
    component = declare_component("lettuce_farm_geolocation", path=str(Path(__file__).parent))
    return component(key=key, default=None)
