-- Vehicle Fault App - PostgreSQL setup
--
-- OPTION A - Easiest (macOS/Homebrew: your user often has create DB rights):
--   From project root:
--   createdb vehicle_monitoring
--   Then ensure .env has DB_NAME=vehicle_monitoring and DB_USER=<your_username>
--
-- OPTION B - Run this SQL as a superuser:
--   psql -d postgres -f scripts/setup_db.sql
--   (or: psql -U postgres -d postgres -f scripts/setup_db.sql)
--
-- The app will create the "predictions" table automatically on first run.

CREATE DATABASE vehicle_monitoring;
