import yfinance as yf

# Get the daily closing prices for the S&P 500 index from 2010 to 2020
data = yf.download("^GSPC", start="2010-01-01", end="2020-12-31")

data.to_csv("SPY_download.csv")
# Print the first few rows of the data
print(data.head())