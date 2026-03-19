CREATE TABLE rain_gage_data (
    site_name TEXT,
    unique_record_id TEXT PRIMARY KEY,
    collect_date TIMESTAMP,
    site_id BIGINT,
    device_id TEXT,
    rain_amount FLOAT,
    rain_intensity FLOAT,
    rssi INT,
    battery_voltage FLOAT,
    geopoint TEXT
);

CREATE TABLE stream_gage_data (
    unique_record_id TEXT PRIMARY KEY,
    collect_date TIMESTAMP,
    site_id BIGINT,
    site_name TEXT,
    device_id TEXT,
    risk_rating TEXT,
    stage FLOAT,
    navd88 FLOAT,
    air_temperature FLOAT,
    water_temperature FLOAT,
    barometric_pressure FLOAT,
    rssi FLOAT,
    battery_voltage FLOAT,
    photo_link TEXT,
    geopoint TEXT
);