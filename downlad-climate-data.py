import cdsapi

c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api/v2',
                  key='83664:e327ac3c-7985-4984-aefa-1411551fc69c')
        
        
result = c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': 'total_precipitation',
        'year': [
            '1981'
        ],
        'month': [
            '01'
        ],
        'day': [
            '01'
        ],
        'time': '14:00',
        'format': 'grib',
        'area': [
            50, 51, 27,
            44,
        ],
    },
    'download.grib')


print(type(result))
