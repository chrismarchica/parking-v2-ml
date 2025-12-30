-- SQL queries for building feature aggregations in PostgreSQL
-- These can be used to pre-compute features for faster training
-- Run these AFTER backfilling the parking_ticket table

-- ============================================================================
-- Location risk score: historical ticket density by grid cell
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS location_risk_scores AS
SELECT
    ROUND(ST_Y(geom::geometry)::numeric, 3) as lat_grid,
    ROUND(ST_X(geom::geometry)::numeric, 3) as lon_grid,
    COUNT(*) as total_tickets,
    COUNT(DISTINCT violation_code) as violation_types,
    COUNT(DISTINCT issue_date) as active_days,
    COUNT(*)::float / NULLIF(COUNT(DISTINCT issue_date), 0) as avg_daily_tickets,
    AVG(fine_amount) as avg_fine,
    MODE() WITHIN GROUP (ORDER BY violation_code) as most_common_violation
FROM parking_ticket
WHERE geom IS NOT NULL
GROUP BY lat_grid, lon_grid;

CREATE INDEX IF NOT EXISTS idx_location_risk_latlon
ON location_risk_scores (lat_grid, lon_grid);


-- ============================================================================
-- Temporal patterns: ticket counts by hour and day of week
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS temporal_patterns AS
SELECT
    EXTRACT(DOW FROM issue_date)::int as day_of_week,
    CASE
        WHEN violation_time ~ '^[0-9]{2}[0-9]{2}[AP]$' THEN
            CASE
                WHEN RIGHT(violation_time, 1) = 'A' THEN
                    CASE
                        WHEN LEFT(violation_time, 2)::int = 12 THEN 0
                        ELSE LEFT(violation_time, 2)::int
                    END
                ELSE
                    CASE
                        WHEN LEFT(violation_time, 2)::int = 12 THEN 12
                        ELSE LEFT(violation_time, 2)::int + 12
                    END
            END
        ELSE NULL
    END as hour_of_day,
    precinct,
    COUNT(*) as ticket_count,
    AVG(fine_amount) as avg_fine
FROM parking_ticket
WHERE violation_time IS NOT NULL
GROUP BY day_of_week, hour_of_day, precinct
HAVING COUNT(*) >= 10;

CREATE INDEX IF NOT EXISTS idx_temporal_patterns_lookup
ON temporal_patterns (day_of_week, hour_of_day, precinct);


-- ============================================================================
-- Precinct statistics
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS precinct_stats AS
SELECT
    precinct,
    county,
    COUNT(*) as total_tickets,
    COUNT(DISTINCT violation_code) as unique_violations,
    COUNT(DISTINCT issuing_agency) as agencies_active,
    MIN(issue_date) as first_ticket_date,
    MAX(issue_date) as last_ticket_date,
    AVG(fine_amount) as avg_fine,
    SUM(fine_amount) as total_fines,
    -- Centroid of geocoded tickets
    AVG(ST_Y(geom::geometry)) as centroid_lat,
    AVG(ST_X(geom::geometry)) as centroid_lon
FROM parking_ticket
WHERE precinct IS NOT NULL
GROUP BY precinct, county;

CREATE INDEX IF NOT EXISTS idx_precinct_stats_lookup
ON precinct_stats (precinct);


-- ============================================================================
-- Street-level aggregations
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS street_stats AS
SELECT
    UPPER(TRIM(street_name)) as street_name_clean,
    precinct,
    county,
    COUNT(*) as ticket_count,
    COUNT(DISTINCT violation_code) as violation_types,
    AVG(fine_amount) as avg_fine,
    MODE() WITHIN GROUP (ORDER BY violation_code) as most_common_violation
FROM parking_ticket
WHERE street_name IS NOT NULL
  AND LENGTH(TRIM(street_name)) > 0
GROUP BY street_name_clean, precinct, county
HAVING COUNT(*) >= 50;

CREATE INDEX IF NOT EXISTS idx_street_stats_lookup
ON street_stats (street_name_clean, precinct);


-- ============================================================================
-- Violation code statistics
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS violation_stats AS
SELECT
    violation_code,
    violation_desc,
    COUNT(*) as total_tickets,
    COUNT(DISTINCT precinct) as precincts_affected,
    COUNT(DISTINCT county) as counties_affected,
    MIN(fine_amount) as min_fine,
    AVG(fine_amount) as avg_fine,
    MAX(fine_amount) as max_fine,
    MIN(issue_date) as first_seen,
    MAX(issue_date) as last_seen
FROM parking_ticket
GROUP BY violation_code, violation_desc
ORDER BY total_tickets DESC;

CREATE INDEX IF NOT EXISTS idx_violation_stats_code
ON violation_stats (violation_code);


-- ============================================================================
-- Monthly trends for seasonality features
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_trends AS
SELECT
    DATE_TRUNC('month', issue_date) as month,
    precinct,
    violation_code,
    COUNT(*) as ticket_count,
    SUM(fine_amount) as total_fines
FROM parking_ticket
WHERE issue_date IS NOT NULL
GROUP BY month, precinct, violation_code;

CREATE INDEX IF NOT EXISTS idx_monthly_trends_lookup
ON monthly_trends (month, precinct);


-- ============================================================================
-- Agency activity patterns
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS agency_stats AS
SELECT
    issuing_agency,
    county,
    COUNT(*) as total_tickets,
    COUNT(DISTINCT violation_code) as violation_types,
    COUNT(DISTINCT precinct) as precincts_covered,
    AVG(fine_amount) as avg_fine,
    MODE() WITHIN GROUP (ORDER BY violation_code) as most_issued_violation
FROM parking_ticket
WHERE issuing_agency IS NOT NULL
GROUP BY issuing_agency, county
ORDER BY total_tickets DESC;


-- ============================================================================
-- Refresh all materialized views (run after data updates)
-- ============================================================================
-- REFRESH MATERIALIZED VIEW CONCURRENTLY location_risk_scores;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY temporal_patterns;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY precinct_stats;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY street_stats;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY violation_stats;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_trends;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY agency_stats;
