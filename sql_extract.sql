-- Create user_records_db database--  
CREATE DATABASE user_records_db;
USE user_records_db;
-- Create sql_extract table-- 
DROP TABLE IF EXISTS sql_extract;

CREATE TABLE sql_extract (
    id INT,
    user_id INT,
    file VARCHAR(255),
    ex_date DATETIME
);
-- confirm import-- 
SELECT * FROM sql_extract limit 5;

-- 1. total records and dinstint users--
SELECT 
    COUNT(*) AS total_records,
    COUNT(DISTINCT user_id) AS distinct_users
FROM sql_extract;

-- 2. latest record per user based on ex_date
SELECT se.id, se.user_id, se.file, se.ex_date
FROM sql_extract se
INNER JOIN (
    SELECT user_id, MAX(ex_date) AS max_date
    FROM sql_extract
    GROUP BY user_id
) latest
ON se.user_id = latest.user_id AND se.ex_date = latest.max_date;


-- 3. Top 5 users with the most records
SELECT 
    user_id,
    COUNT(*) AS record_count
FROM sql_extract
GROUP BY user_id
ORDER BY record_count DESC
LIMIT 5;

-- 4. Total records per day
SELECT 
    DATE(ex_date) AS day,
    COUNT(*) AS records_per_day
FROM sql_extract
GROUP BY DATE(ex_date)
ORDER BY day;
