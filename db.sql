-- create user for smartlight
CREATE USER smartlight WITH ENCRYPTED PASSWORD 'lampupintar';
CREATE DATABASE smartlight_db;
GRANT ALL PRIVILEGES ON DATABASE smartlight_db TO smartlight;
ALTER USER smartlight WITH SUPERUSER CREATEDB CREATEROLE;


CREATE TABLE sl_vision(
	id SERIAL PRIMARY KEY,
	id_cam INTEGER, 
	id_dot INTEGER,
	people_count INTEGER,
	time_vision TIMESTAMP
);