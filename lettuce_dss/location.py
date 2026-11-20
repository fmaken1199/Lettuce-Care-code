"""Country-filtered place lookup. An unavailable search has no default location."""
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

def search_pakistan_places(query: str, timeout: float = 5.0) -> list[dict]:
    query = query.strip()
    if len(query) < 3:
        return []
    url = 'https://geocoding-api.open-meteo.com/v1/search?' + urlencode(
        {'name': query, 'count': 10, 'language': 'en', 'format': 'json', 'countryCode': 'PK'})
    with urlopen(Request(url, headers={'User-Agent':'Pakistan-Lettuce-DSS/1.0'}), timeout=timeout) as response:
        data = json.load(response)
    return [{'latitude':float(r['latitude']), 'longitude':float(r['longitude']),
             'label':', '.join(x for x in [r.get('name'),r.get('admin1'),'Pakistan'] if x),
             'source':'Open-Meteo place search'} for r in data.get('results',[]) if r.get('country_code') == 'PK']
