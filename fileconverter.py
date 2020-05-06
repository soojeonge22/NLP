import pandas as pd

in_file =(r'/home/soojeong/soojeonge22/web-crawler/kowiki/kowiki_20200429.csv')
out_file = (r'/home/soojeong/soojeonge22/web-crawler/kowiki/kowiki.txt')
SEPARATOR = u"\u241D"
df = pd.read_csv(in_file, sep=SEPARATOR, engine="python")
with open(out_file, "w") as f:
  for index, row in df.iterrows():
    f.write(row["text"]) # title 과 text를 중복 되므로 text만 저장 함
    f.write("\n\n\n\n") # 구분자
