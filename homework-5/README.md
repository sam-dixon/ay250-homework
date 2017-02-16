# HW 5

`populate_db.ipynb` initializes the database by merging the `hw_5_data/ICAO_airports.csv` and `hw_5_data/top_airports.csv`. It also includes a webscraper that populates the database with some interesting weather information from [Weather Underground](www.wunderground.com).

`analysis.ipynb` creates a new table in the database that calculates the correlation coefficients for one-day changes in high temperature and inches of precipitation between each pair of the top 50 airports. There are also some plots that show some interesting patterns in the data.

The database that is created in both of these notebooks is already included in the repository as `airports.db`.