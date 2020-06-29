from plot_curves_window import curves as curves_window
from plot_curves_window_trend import curves as curves_window_trend
from plot_window_germany import curves as germany_map
from plot_interaction_kernel_window import interaction_kernel
from shared_utils import make_county_dict

start = 89 
county_dict = make_county_dict()

for c in county_dict.keys():
    print(c)
    curves_window(start, c, n_weeks=3, model_i=35, save_plot=True)
    curves_window_trend(start, c, save_plot=True)
    germany_map(start, save_plot=True)
    interaction_kernel(start, save_plot=True)
