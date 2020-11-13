
import csv
from utils import load_hotspots
import random
import folium
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import h3



def get_hex_limit_info(fn):

    with open(fn, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        reader.__next__()
        count = 0
        results = []
        for row in reader:
            count += 1
            results.append(dict(hex=row[0], res=int(row[1]), limit=int(row[2]), unclipped=int(row[4])))


    return results

def map_hex_limits(hex_info, res=8, outfile=None):


    bbox = [48.922499,-124.872527, 24.806681, -66.896522]
    lat = (bbox[0] + bbox[2])/2
    lng = (bbox[1] + bbox[3])/2
    tiles = 'http://{s}.tiles.mapbox.com/v4/wtgeographer.2fb7fc73/{z}/{x}/{y}.png?access_token=pk.eyJ1IjoiY2Fybml2ZXJvdXMxOSIsImEiOiJja2U5Y3RyeGsxejd1MnBxZ2RiZXUxNHE2In0.S_Ql9KARjRdzgh1ZaJ-_Hw'
    my_map = folium.Map(location=[lat, lng], zoom_start=6,
                        #tiles=tiles,
                        #API_key='pk.eyJ1IjoiY2Fybml2ZXJvdXMxOSIsImEiOiJja2U5Y3RyeGsxejd1MnBxZ2RiZXUxNHE2In0.S_Ql9KARjRdzgh1ZaJ-_Hw'
                        #attr='Mapbox'
                        )
    tgt_res_info = []
    for x in hex_info:
        if x['res'] != res:
            continue
        tgt_res_info.append(x)

    vals = [x['limit'] for x in tgt_res_info]


    cnorm = mpl.colors.Normalize(vmin=np.min(vals), vmax=np.max(vals))# np.max(vals))
    colmap = cm.ScalarMappable(norm=cnorm, cmap='RdYlGn')

    parent_hexs = set()
    for row in tgt_res_info:
        hex = row['hex']
        parent_hexs.add(h3.h3_to_parent(hex, res-1))
        color = "#" + ''.join([f"{x:02x}" for x in colmap.to_rgba(row['limit'], bytes=True)[:3]])
        #print(f"lim:{row['limit']}, color:{color}")

        hex_points = list(h3.h3_to_geo_boundary(hex, False))
        hex_points.append(hex_points[0])
        folium.Polygon(hex_points, weight=1.5, color='black', opacity=.45, fill_color=color, popup=f"limit:{row['limit']}, #Hs:{row['unclipped']}", fill=True, fill_opacity=0.8,).add_to(my_map)

    for h in parent_hexs:
        hex_points = list(h3.h3_to_geo_boundary(h, False))
        hex_points.append(hex_points[0])
        folium.PolyLine(hex_points, weight=1.5, color='black', opacity=.45).add_to(my_map)
    my_map.save(outfile)

def main(R=8):
    fn = f"hexDensities_RewardScale_R{R}.csv"
    hex_info = get_hex_limit_info(fn)
    map_res = 8
    map_hex_limits(hex_info, res=map_res, outfile=f"hex_res{map_res}_limits.html")



if __name__ == '__main__':
    main()
