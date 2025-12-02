## 📌 Overview

This repository contains **end-to-end data analysis in Google BigQuery** using a global **Sports Leagues Dataset**, including:

- Aggregate SQL queries  
- Window functions to rank leagues  
- Visualization using BigQuery charts  
- Exporting results to Sheets / CSV  

**BigQuery Table:** `mlopsl.mlo.lab6_table`

---

## 🧮 SQL Queries

### Query 1 — Top 5 Countries by Revenue & Viewership

```sql
SELECT
  Country,
  COUNT(*) AS num_leagues,
  SUM(Revenue_USD) AS total_revenue_usd,
  AVG(Viewership) AS avg_viewership
FROM `mlopsl.mlo.lab6_table`
GROUP BY Country
HAVING SUM(Revenue_USD) IS NOT NULL
ORDER BY total_revenue_usd DESC
LIMIT 5;
```


#### Query 2 — Highest-Revenue League per Sport
```sql
WITH ranked AS (
  SELECT
    `League ID` AS league_id,
    `League Name` AS league_name,
    Sport,
    Country,
    Revenue_USD,
    RANK() OVER (
      PARTITION BY Sport
      ORDER BY Revenue_USD DESC
    ) AS rev_rank_in_sport
  FROM `mlopsl.mlo.lab6_table`
)
SELECT
  league_id,
  league_name,
  Sport,
  Country,
  Revenue_USD
FROM ranked
WHERE rev_rank_in_sport = 1
ORDER BY Revenue_USD DESC;
```

📊 Visualizations

Visuals created in BigQuery:
📈 Pie Chart — Country share of global viewership
📉 Bar Chart — Number of leagues per country
📊 Horizontal Bar — Total viewership across countries
