import argparse
import pandas as pd

from plot_window_germany import curves as map_csv # I'M SO SORRY; BRANCH::SIMPLER FIXES THIS

def main(begin, end):
    for start in range(begin, end+1):
        map_csv(start, n_weeks=3, model_i=35, save_csv=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrects previously falsely \
        assigned data in map.csv files by calling the corrected method from start_id \
        to end_id")
    
    parser.add_argument("start_id", nargs=1, type=int)
    parser.add_argument("end_id", nargs=1, type=int)
    args = parser.parse_args()

    start_day1 = pd.Timestamp('2020-01-28') + pd.Timedelta(days=args.start_id[0])
    start_day2 = pd.Timestamp('2020-01-28') + pd.Timedelta(days=args.end_id[0])

    # add training and nowcast period, substract offset
    ref_day1 = start_day1 + pd.Timedelta(days=21+5) - pd.Timedelta(days=1)
    ref_day2 = start_day2 + pd.Timedelta(days=21+5) - pd.Timedelta(days=1)

    print("Correcting Map.csv from {} to {} inclusively \n \
           This corresponds to dates {} to {}".format(args.start_id[0], 
                                                      args.end_id[0],
                                                      ref_day1,
                                                      ref_day2))
    main(args.start_id[0], args.end_id[0])
