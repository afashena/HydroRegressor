# This contains useful helper functions for the Streamlit UI.

def parse_site_ids(site_id_str: str):
    """Parse a newline-separated string of site IDs into a list."""
    return [site_id.strip() for site_id in site_id_str.split("\n") if site_id.strip()]