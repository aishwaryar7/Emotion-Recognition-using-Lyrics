import pandas as pd
import pandas_profiling

def main():
    df = pd.read_excel("ml_lyrics.xlsx", parse_dates=True, encoding='latin-1')
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(outputfile="PandasProfiling.html")

if __name__ == "__main__":
    main()
