
import csv
import utils
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError:
    print("-W- matplotlib or numpy cannot be imported, file export only")
from h3 import h3



def is_interactive(hspot):
    # over-simplified but conservative.  Filters out dead hotspots but should look at witness list to determine
    # if a hotspot is "interactive"
    return hspot.get('location') is not None and hspot['status']['online'] == 'online'


def sample(hotspots, density_tgt, density_max, R, N):


    # ==============================================================
    # Part 1, find hexs and density of hexs containing interactive
    #         hotspots at target resolution
    # ==============================================================

    # determine density of occupied "tgt_resolution" hexs.  This sets our initial conditions.  I also track "actual" vs
    # clipped density to find discrepancies
    #hex_density will be keys of hexs (all resolutions) with a value of dict(clipped=0, actual=0)
    hex_density = dict()
    for h in hotspots:
        if is_interactive(h):
            hex = h3.h3_to_parent(h['location'], R)

            # initialize the hex if not in dictionary
            if hex not in hex_density:
                hex_density[hex] = dict(clipped=0, actual=0)
            hex_density[hex]['clipped'] += 1
            hex_density[hex]['actual'] += 1

    print(f"{len(hex_density)} unique res {R} hexs")

    #build a set of R resolution hexs, occupied child hexs are how we build occupied hexs for parent levels
    child_hexs = set(hex_density.keys())

    # ==============================================================
    # Part 2, go from high to low res, clipping density and determining
    #         densities of parent hexs
    # ==============================================================

    # iterate through resultion from just above target to 1 clipping child densities and calculating appropriate hex
    # densities at "resolution"
    for resolution in range(R - 1, 0, -1):
        # hold set of hex's to evaluate
        occupied_hexs = dict() # key = parent hex, values = list of child hexs
        # density target and limit at  child's resolution.  This is simply scaled up by increased area
        density = density_tgt * 7**(R-resolution-1)
        density_limit = density_max * 7**(R-resolution-1)

        # print(f"res: {resolution+1}, density: {density}, limit: {density_limit}")

        # 1. find all occupied hexs at this resolution based on child hexs
        for h in child_hexs:
            occupied_hexs.setdefault(h3.h3_to_parent(h, resolution), [])
            occupied_hexs[h3.h3_to_parent(h, resolution)].append(h)

        # for each occupied hex at this level, evaluate its children
        for h in occupied_hexs:
            children = occupied_hexs[h]
            # 1. find count of children > tgt_density to possibly elevate clipping value of N threshold met.
            above_density_cnt = 0
            for c in children:
                if hex_density.get(c, dict(clipped=0, actual=0))['clipped'] >= density:
                    above_density_cnt += 1

            hex_raw_density = 0
            hex_unclipped_density = 0
            # clip children at density_tgt unless above_density_cnt meets threshold, then calculate appropriate clipping
            clip = density
            if above_density_cnt > N:
                clip = min(density_limit, density * (above_density_cnt - N + 1))

            # iterate through all children clipping density and calculating density for this hex.  Note this may not be
            # appropriately clipped since we need to evaluate all this hex's siblings (will be done in next iteration)
            # of outer loop
            for c in children:
                hex_density[c]['clipped'] = min(clip, hex_density[c]['clipped'])
                hex_unclipped_density += hex_density[c]['actual']
                hex_raw_density += hex_density[c]['clipped']

            # set this hex raw density unclipped (will be clipped at parent)
            hex_density[h] = dict(clipped=hex_raw_density, actual=hex_unclipped_density)

        print(f"total of {len(occupied_hexs)} occupied hexes at resolution {resolution}")
        # occupied hex's at this resolution are child hexs in next resolution
        child_hexs = occupied_hexs

    # ==============================================================
    # Part 3, print / store analysis
    # ==============================================================

    # occupied_hex's is now the top level hex evaluated.  Start here for descending to target a hotspot
    top_count = 0
    for h in occupied_hexs:
        #print(f"hex {h} has density {hex_density[h]}")
        top_count += hex_density[h]['clipped']

    print(f"total density of all top level hexs = {top_count}")
    # for k in hex_density.keys():
    #     hex_density[k]['border'] = h3.h3_to_geo_boundary(k, False)

    # track max/min hex for gut check
    max_prob = 0
    min_prob = 1
    max_h = None
    min_h = None
    interactive_hspots = 0

    with open(f'hex_occupancy_R{R}_N{N}_tgt{density_tgt}_max{density_max}.csv', 'w', newline='') as csvfile:
        hex_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hex_writer.writerow(['hex', 'resolution', 'density_clipped', 'density_actual'])
        for k in hex_density:
            hex_writer.writerow([k, h3.h3_get_resolution(k), hex_density[k]['clipped'], hex_density[k]['actual']])

    with open(f'hotspot_tgting_prob_R{R}_N{N}_tgt{density_tgt}_max{density_max}.csv', 'w', newline='') as csvfile:
        hspot_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hspot_writer.writerow(['address', 'name', 'city', 'state', 'probability'])
        # iterate through all interactive hotspots and evaluate probability of targeting. this will be outputted to CSV
        for hspot in hotspots:
            # start at top level and iterate through determining odds of selection
            if not is_interactive(hspot):
                continue
            interactive_hspots += 1
            sibling_total = top_count
            sibling_unclipped = 0
            probability = 1
            for res in range(1, R+1):
                hex = h3.h3_to_parent(hspot['location'], res)
                probability *= hex_density[hex]['clipped'] / sibling_total
                sibling_total = hex_density[hex]['clipped']
                sibling_unclipped = hex_density[hex]['actual']

            # for determining individual hotspot probability we consider all hotspots its hex regardless of clipping
            probability = probability / sibling_unclipped
            if probability > max_prob:
                max_h = hspot
                max_prob = probability
            elif probability < min_prob:
                min_h = hspot
                min_prob = probability
            hspot_writer.writerow([hspot['address'], hspot['name'], hspot['geocode']['short_city'], hspot['geocode']['short_state'], probability])
            # print(f"hotspot {hspot['name']:30} has {sibling_unclipped} hotspots in res8 cell, probability {probability*100:.8f}%")

        print(f"total of {interactive_hspots} interactive hotspots")
        print(f"Most likely hotspot is  {max_h['name']:30} with 1:{1/max_prob:.0f} odds of targeting")
        print(f"Least likely hotspot is {min_h['name']:30} with 1:{1/min_prob:.0f} odds of targeting")


def odds_by_city(file='hotspot_tgting_prob_R8_N2_tgt1_max4.csv'):
    hspots = 0
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        city_dict = dict()
        spamreader.__next__()
        for row in spamreader:
            city = f"{row[2]}, {row[3]}"
            city_dict.setdefault(city, [])
            city_dict[city].append(float(row[-1]))
            hspots += 1
        axticks = []
        mymax = 0
        for x in list(city_dict.keys()):
            if len(city_dict[x]) < 10:
                city_dict.pop(x)
                continue
            city_dict[x] = np.array(city_dict[x]) * hspots

            if len(city_dict[x]) >= 50 or np.mean(city_dict[x]) <= 0.6:
                mymax = max(mymax, np.max(city_dict[x]))
                axticks.append(f"{x} [{len(city_dict[x]):3d}]")
            else:
                city_dict.pop(x)
        plt.violinplot(
            list(city_dict.values()),
            widths=1.5,
            points=100,
            bw_method=.05,
            showmeans=True,
            vert=False,
            #showextrema=True
        )
        plt.yticks(range(1, 1+len(axticks)), axticks)
        plt.ylim([0, len(axticks)+1])
        plt.xlim([0, mymax * 1.02])
        plt.xlabel("normalized targeting probability (vs 1/|hotspots|)")
        plt.title(f"violin plot of targeting probabilities by city\n{file}")
        plt.grid()
        plt.gcf().set_size_inches(10, 10)
        plt.tight_layout()

        plt.show()

def hex_occupancy(hexs, level=None):
    """
    converts list of hexs to level (if provided) and returns a dictionary of with keys of hexs and values of counts
    :param hexs: list of hexs
    :param level: desired level (all provided hex's should be below or equal to level
    :return:
    """
    res = dict()
    for h in hexs:
        if level:
            h = h3.h3_to_parent(h, level)
        res.setdefault(h, 0)
        res[h] += 1
    return res

def main():
    hs = utils.load_hotspots()
    sample(
        hotspots=hs,
        density_tgt=1,
        density_max=4,
        R=8,
        N=2
    )

if __name__ == '__main__':
    odds_by_city()