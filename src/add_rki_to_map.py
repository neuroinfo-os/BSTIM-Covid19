import os
import pickle as pkl
import pandas as pd

data = pd.read_csv('../data/diseases/covid19.csv', sep=',', encoding='iso-8859-1', index_col=0)
data.index = [pd.Timestamp(date) for date in data.index]

with open('../data/counties/counties.pkl', 'rb') as f:
    counties = pkl.load(f)

shared_assets = '/p/project/covid19dynstat/shared_assets/'
figures = os.path.join(shared_assets, 'figures/')
dates = next(os.walk(figures))[1]

# errors if the RKI based csv is out of date with the available folders
for date in sorted(dates):

    map_csv_path = os.path.join(figures, date, 'map.csv')   
    if not os.path.exists(map_csv_path):
        continue
    
    day_0 = pd.Timestamp(date.replace('_','-')) + pd.Timedelta(days=25)
    
    map_rki = data.loc[day_0].values.astype('float64')
    for (i, (key, _)) in enumerate(counties.items()):
        n_people = counties[key]['demographics'][('total', 2018)]
        map_rki[i] = map_rki[i] / n_people * 100000
    
    map_data = pd.read_csv(map_csv_path, index_col=0)
    map_data["newInf100k_RKI"] = list(map_rki)

    map_data.to_csv(map_csv_path)
