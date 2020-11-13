
import csv
from utils import load_hotspots
import random
import folium
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import h3


def get_hotspot_probs(R, N, tgt, max_density):
    fn = f"hotspot_tgting_prob_R{R}_N{N}_tgt{tgt}_max{max_density}.csv"
    with open(fn, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        reader.__next__()
        count = 0
        results = []
        for row in reader:
            count += 1
            results.append(dict(addr=row[0], name=row[1], odds=float(row[-1])))


    for r in results:
        r['odds'] = r['odds'] * count

    return results


def get_hotspot_scales(R, N, tgt, max_density, fn=None):
    if fn is None:
        fn = f"hotspot_RewardScale_R{R}_N{N}_tgt{tgt}_max{max_density}.csv"
    with open(fn, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        reader.__next__()
        count = 0
        results = []
        for row in reader:
            count += 1
            results.append(dict(addr=row[0], odds=float(row[-1])))


    return results

def plot_hotspot_probs(hprobs, lat=None, lng=None, R=8, geo_range=1.0, outputfile='hotspot_probs.html', bbox=None):
    """

    :param hprobs: list of hotspot probabilities
    :param lat: map center latitude
    :param lng: map center longitude
    :param outputfile:
    :param bbox: lat upper left, long upper left, lat upper right, long upper right
    :return:
    """
    hs = load_hotspots()
    h_by_addr = dict()
    for h in hs:
        h_by_addr[h['address']] = h

    if not bbox:
        bbox = [lat + geo_range, lng - geo_range, lat - geo_range, lng + geo_range]
    else:
        lat = (bbox[0] + bbox[2])/2
        lng = (bbox[1] + bbox[3])/2

    my_map = folium.Map(location=[lat, lng], zoom_start=6)

    vals = [x['odds'] for x in hprobs]

    cnorm = mpl.colors.TwoSlopeNorm(vcenter=1.0, vmin=np.min(vals), vmax=2)# np.max(vals))
    colmap = cm.ScalarMappable(norm=cnorm, cmap='RdYlGn')

    idc = np.argsort(vals)
    hexs = set([])  # store hex's where odds < 1.0 for displaying
    hex_parent = set([])
    hex_gparent = set([])

    for idx in idc[::-1]:
        hp = hprobs[idx]
        hlat = h_by_addr[hp['addr']]['lat'] + (random.random()-0.5)*.0004
        hlng = h_by_addr[hp['addr']]['lng'] + (random.random()-0.5)*.0004

        if not (bbox[2] < hlat < bbox[0] and bbox[1] < hlng < bbox[3]):
            continue

        color = "#" + ''.join([f"{x:02x}" for x in colmap.to_rgba(hp['odds'], bytes=True)[:3]])

        folium.CircleMarker(
            (hlat, hlng), color='black', fill_color=color, popup=f"{h_by_addr[hp['addr']]['name']} [{hp['odds']:.2f}]", fill=True, fill_opacity=0.9,
            number_of_sides=8, radius=11, opacity=.35, weight=2, z_index_offset=2-int(hp['odds'])
        ).add_to(my_map)


        hexs.add(
            h3.h3_to_parent(h_by_addr[hp['addr']]['location'], R)
        )
        hex_parent.add(
            h3.h3_to_parent(h_by_addr[hp['addr']]['location'], R-1)
        )
        hex_gparent.add(
            h3.h3_to_parent(h_by_addr[hp['addr']]['location'], R-2)
        )

    print(f"drawing {len(hexs)} target hexs")
    for hex in hexs:
        hex_points = list(h3.h3_to_geo_boundary(hex, False))
        hex_points.append(hex_points[0])
        folium.PolyLine(hex_points, weight=1.5, color='black', opacity=.45).add_to(my_map)

    print(f"drawing {len(hex_parent)} parent hexs")
    for hex in hex_parent:
        hex_points = list(h3.h3_to_geo_boundary(hex, False))
        hex_points.append(hex_points[0])
        folium.PolyLine(hex_points, weight=1.5, opacity=.65).add_to(my_map)
    print(f"drawing {len(hex_gparent)} grandparent hexs")
    for hex in hex_gparent:
        hex_points = list(h3.h3_to_geo_boundary(hex, False))
        hex_points.append(hex_points[0])
        folium.PolyLine(hex_points, weight=1.5, color='white', opacity=.65).add_to(my_map)

    my_map.save(outputfile)


def map_hotspot_scale(hscale, lat=None, lng=None, R=8, geo_range=1.0, outputfile='hotspot_probs.html', bbox=None):
    """

    :param hscale: list of hotspot probabilities
    :param lat: map center latitude
    :param lng: map center longitude
    :param outputfile:
    :param bbox: lat upper left, long upper left, lat upper right, long upper right
    :return:
    """
    hs = load_hotspots()
    h_by_addr = dict()
    for h in hs:
        h_by_addr[h['address']] = h

    if not bbox:
        bbox = [lat + geo_range, lng - geo_range, lat - geo_range, lng + geo_range]
    else:
        lat = (bbox[0] + bbox[2])/2
        lng = (bbox[1] + bbox[3])/2
    tiles = 'http://{s}.tiles.mapbox.com/v4/wtgeographer.2fb7fc73/{z}/{x}/{y}.png?access_token=pk.eyJ1IjoiY2Fybml2ZXJvdXMxOSIsImEiOiJja2U5Y3RyeGsxejd1MnBxZ2RiZXUxNHE2In0.S_Ql9KARjRdzgh1ZaJ-_Hw'
    my_map = folium.Map(location=[lat, lng], zoom_start=6,
                        #tiles=tiles,
                        #API_key='pk.eyJ1IjoiY2Fybml2ZXJvdXMxOSIsImEiOiJja2U5Y3RyeGsxejd1MnBxZ2RiZXUxNHE2In0.S_Ql9KARjRdzgh1ZaJ-_Hw'
                        #attr='Mapbox'
                        )

    vals = [x['odds'] for x in hscale]
    avg = np.mean(vals)
    #vals = np.array(vals)/avg
    print(f"{np.max(vals)}")
    print(f"average scaling factor = {avg}")


    cnorm = mpl.colors.TwoSlopeNorm(vcenter=avg, vmin=np.min(vals), vmax=np.max(vals)*1.2)
    colmap = cm.ScalarMappable(norm=cnorm, cmap='RdYlGn')

    idc = np.argsort(vals)
    hexs = set([])  # store hex's where odds < 1.0 for displaying
    hex_parent = set([])
    hex_gparent = set([])
    hex_ggparent = set([])

    for idx in idc[::-1]:
        hp = hscale[idx]
        hlat = h_by_addr[hp['addr']]['lat'] #+ (random.random()-0.5)*.0000
        hlng = h_by_addr[hp['addr']]['lng'] #+ (random.random()-0.5)*.0000

        if not (bbox[2] < hlat < bbox[0] and bbox[1] < hlng < bbox[3]):
            continue

        color = "#" + ''.join([f"{x:02x}" for x in colmap.to_rgba(hp['odds'], bytes=True)[:3]])

        folium.CircleMarker(
            (hlat, hlng), color='black', fill_color=color, popup=f"{h_by_addr[hp['addr']]['name']} [{hp['odds']:.2f}]", fill=True, fill_opacity=0.9,
            number_of_sides=8, radius=11, opacity=.35, weight=2, z_index_offset=2-int(hp['odds'])
        ).add_to(my_map)


        hexs.add(
            h3.h3_to_parent(h_by_addr[hp['addr']]['location'], R)
        )
        hex_parent.add(
            h3.h3_to_parent(h_by_addr[hp['addr']]['location'], R-1)
        )
        hex_gparent.add(
            h3.h3_to_parent(h_by_addr[hp['addr']]['location'], R-2)
        )
        hex_ggparent.add(
            h3.h3_to_parent(h_by_addr[hp['addr']]['location'], R-3)
        )

    print(f"drawing {len(hexs)} target hexs")
    for hex in hexs:
        hex_points = list(h3.h3_to_geo_boundary(hex, False))
        hex_points.append(hex_points[0])
        folium.PolyLine(hex_points, weight=1.5, color='black', opacity=.45).add_to(my_map)
        folium.Polygon(hex_points)

    print(f"drawing {len(hex_parent)} parent hexs")
    for hex in hex_parent:
        hex_points = list(h3.h3_to_geo_boundary(hex, False))
        hex_points.append(hex_points[0])
        folium.PolyLine(hex_points, weight=1.5, opacity=.65).add_to(my_map)
    print(f"drawing {len(hex_gparent)} grandparent hexs")
    for hex in hex_gparent:
        hex_points = list(h3.h3_to_geo_boundary(hex, False))
        hex_points.append(hex_points[0])
        folium.PolyLine(hex_points, weight=1.5, color='white', opacity=.65).add_to(my_map)
    print(f"drawing {len(hex_gparent)} great grandparent hexs")
    for hex in hex_ggparent:
        hex_points = list(h3.h3_to_geo_boundary(hex, False))
        hex_points.append(hex_points[0])
        folium.PolyLine(hex_points, weight=1.5, color='pink', opacity=.65).add_to(my_map)

    my_map.save(outputfile)


def main():
    SF = [37.692514, -122.236160]
    Modesto = [37.6393, -120.9970]
    NYC = [40.7831, -73.9712]
    minneapolis = [44.9778, -93.2650]
    denver = [39.7392, -104.9903]
    LA = [34.0522, -118.2437]
    Miami = [25.7617, -80.1918]
    CapeMay = [39.442557, -75.018516]

    USA_bbox = [48.922499,-124.872527, 24.806681, -66.896522]

    hprobs = get_hotspot_scales(R=9, N=2, tgt=1, max_density=4, fn="hotspot_RewardScale_R9.csv")
    # plot_hotspot_probs(hprobs, SF[0], SF[1], outputfile='para1_geohip_SF.html')
    # plot_hotspot_probs(hprobs, Modesto[0], Modesto[1], outputfile='para1_geohip_Modesto.html')
    # plot_hotspot_probs(hprobs, NYC[0], NYC[1], outputfile='para1_geohip_NYC.html')
    # plot_hotspot_probs(hprobs, minneapolis[0], minneapolis[1], outputfile='para1_geohip_Minneapolis.html')
    # plot_hotspot_probs(hprobs, denver[0], denver[1], outputfile='para1_geohip_Denver.html')
    # plot_hotspot_probs(hprobs, LA[0], LA[1], outputfile='para1_geohip_LA.html')
    # plot_hotspot_probs(hprobs, Miami[0], Miami[1], outputfile='para1_geohip_Miami.html')
    # plot_hotspot_probs(hprobs, CapeMay[0], CapeMay[1], outputfile='para1_geohip_CapeMayNJ.html')
    map_hotspot_scale(hprobs, bbox=USA_bbox, R=9, outputfile='para1_geohip_USA.html')

if __name__ == '__main__':
    main()

