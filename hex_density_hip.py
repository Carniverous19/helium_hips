



import csv
import utils
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError:
    print("-W- matplotlib or numpy cannot be imported, file export only")
from h3 import h3

interactives = set([])
with open('witnesses.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    count = 0
    results = []
    for row in reader:
        interactives.update(row)
with open('witnesses2.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    count = 0
    results = []
    for row in reader:
        interactives.update(row)

lone_wolfs = 0

def is_interactive(hspot):
    global lone_wolfs
    global interactives
    # over-simplified but conservative.  Filters out dead hotspots but should look at witness list to determine
    # if a hotspot is "interactive"
    if not (hspot.get('location') is not None and hspot['geocode']['short_country'] == 'US' and hspot['status']['online'] == 'online'):
        return False
    if hspot['address'] not in interactives:
        #print(f"{hspot['name']} has location but not interactive, probably lone wolf")
        lone_wolfs += 1
        return False
    return True


def occupied_neighbors(hex, density_tgt, density_max, N, hex_density, method='siblings'):
    """

    :param hex: hex to query
    :param density_tgt: target density for hexs at this resolution
    :param density_max: maximum density at this resolution
    :param hex_density: dictionary of densities at each hex
    :param N:
    :param method: either siblings or neighbors
    :return:
    """
    # neigbhors = h3.hex_range(h, 1)
    #neigbhors = h3.h3_to_children(h3.h3_to_parent(h, resolution - 1), resolution)
    res = h3.h3_get_resolution(hex)
    if method == 'siblings':
        neighbors = h3.h3_to_children(h3.h3_to_parent(hex, res - 1), res)
    elif method == 'neighbors':
        neighbors = h3.hex_range(hex, 1)


    neighbors_above_tgt = 0
    for n in neighbors:
        if n not in hex_density:
            continue
        if hex_density[n]['clipped'] >= density_tgt:
            neighbors_above_tgt += 1
    clip = min(
        density_max,
        density_tgt * max(1, (neighbors_above_tgt - N + 1))
    )
    return clip



def sample(hotspots, density_tgt, density_max, R, N):


    # ==============================================================
    # Part 1, find hexs and density of hexs containing interactive
    #         hotspots at highest resolution
    # ==============================================================

    # determine density of occupied "tgt_resolution" hexs.  This sets our initial conditions.  I also track "actual" vs
    # clipped density to find discrepancies
    #hex_density will be keys of hexs (all resolutions) with a value of dict(clipped=0, actual=0)
    hex_density = dict()
    interactive = 0
    for h in hotspots:
        if is_interactive(h):
            hex = h3.h3_to_parent(h['location'], R)
            interactive += 1
            # initialize the hex if not in dictionary
            if hex not in hex_density:
                hex_density[hex] = dict(clipped=0, actual=0, unclipped=0)
            hex_density[hex]['clipped'] += 1
            hex_density[hex]['actual'] += 1
            hex_density[hex]['unclipped'] += 1
    print(f"{len(hotspots)} hotspots")
    print(f"{len(hex_density)} unique res {R} hexs")
    print(f"{lone_wolfs} lone wolfs")
    print(f"{interactive} interactive hotspots")
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
                if hex_density.get(c, dict(clipped=0, actual=0, uncipped=0))['clipped'] >= density:
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
            hex_density[h] = dict(clipped=hex_raw_density, actual=hex_unclipped_density, unclipped=hex_raw_density)

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

    # track max/min hex for gut check
    interactive_hspots = 0

    with open(f'hex_occupancy_R{R}_N{N}_tgt{density_tgt}_max{density_max}.csv', 'w', newline='') as csvfile:
        hex_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hex_writer.writerow(['hex', 'resolution', 'density_clipped', 'density_actual'])
        for k in hex_density:
            hex_writer.writerow([k, h3.h3_get_resolution(k), hex_density[k]['clipped'], hex_density[k]['actual']])

    with open(f'hotspot_tgting_prob_R{R}_N{N}_tgt{density_tgt}_max{density_max}.csv', 'w', newline='') as csvfile:
        hspot_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hspot_writer.writerow(['address', 'name', 'city', 'state', 'prob'])
        # iterate through all interactive hotspots and evaluate probability of targeting. this will be outputted to CSV
        for hspot in hotspots:
            # start at top level and iterate through determining odds of selection
            if not is_interactive(hspot):
                continue
            interactive_hspots += 1
            sibling_total = top_count
            sibling_unclipped = 0
            probability = 1
            scale = 1
            for res in range(1, R+1):
            #for res in range(R, 0, -1):
                hex = h3.h3_to_parent(hspot['location'], res)
                prob_orig = probability
                probability *= hex_density[hex]['clipped']/sibling_total
                scale_orig = scale

                scale *= hex_density[hex]['clipped'] / hex_density[hex]['unclipped']
                if hspot['name'] == 'blunt-clay-puppy':
                    print(f"{hex} h3res:{res} has density clipped/unclipped of {hex_density[hex]['clipped']:3d}/{hex_density[hex]['unclipped']:3d}, prob reduced: {prob_orig:.3f} to {probability:.3f}")
                sibling_total = hex_density[hex]['clipped']
                sibling_unclipped = hex_density[hex]['actual']

            probability *= 1/sibling_unclipped

            hspot_writer.writerow([hspot['address'], hspot['name'], hspot['geocode']['short_city'], hspot['geocode']['short_state'], f"{probability:.6f}"])
            # print(f"hotspot {hspot['name']:30} has {sibling_unclipped} hotspots in res8 cell, probability {probability*100:.8f}%")

        print(f"total of {interactive_hspots} interactive hotspots")



def sample_neighbor(hotspots, density_tgt, density_max, R, N):


    # ==============================================================
    # Part 1, find hexs and density of hexs containing interactive
    #         hotspots at target resolution
    # ==============================================================

    # determine density of occupied "tgt_resolution" hexs.  This sets our initial conditions.  I also track "actual" vs
    # clipped density to find discrepancies
    #hex_density will be keys of hexs (all resolutions) with a value of dict(clipped=0, actual=0)
    hex_density = dict()
    interactive = 0
    for h in hotspots:
        if is_interactive(h):
            hex = h3.h3_to_parent(h['location'], R)
            interactive += 1
            # initialize the hex if not in dictionary
            if hex not in hex_density:
                hex_density[hex] = dict(clipped=0, actual=0, unclipped=0)
            hex_density[hex]['clipped'] += 1
            hex_density[hex]['actual'] += 1
            hex_density[hex]['unclipped'] += 1

    for h in hex_density.keys():
        clip = occupied_neighbors(h, density_tgt, density_max, N, hex_density, method='neighbors')

        hex_density[h]['clipped'] = min(hex_density[h]['clipped'], clip)
        hex_density[h]['limit'] = clip

    print(f"{len(hotspots)} hotspots")
    print(f"{len(hex_density)} unique res {R} hexs")
    print(f"{lone_wolfs} lone wolfs")
    print(f"{interactive} interactive hotspots")
    #build a set of R resolution hexs, occupied child hexs are how we build occupied hexs for parent levels
    occupied_higher_res = set(hex_density.keys())

    # ==============================================================
    # Part 2, go from high to low res, clipping density and determining
    #         densities of parent hexs
    # ==============================================================

    # iterate through resultion from just above target to 1 clipping child densities and calculating appropriate hex
    # densities at "resolution"
    for resolution in range(R - 1, 0, -1):
        # hold set of hex's to evaluate
        occupied_hexs = set([]) # key = parent hex, values = list of child hexs
        # density target and limit at  child's resolution.  This is simply scaled up by increased area
        density_res_tgt = density_tgt * 7**(R-resolution)
        density_res_max = density_max * 7**(R-resolution)

        # 1. find all occupied hexs at this resolution based on child hexs
        for h in occupied_higher_res:
            occupied_hexs.add(h3.h3_to_parent(h, resolution))

        for h in occupied_hexs:
            children = h3.h3_to_children(h, resolution + 1)

            # calculate density of this hex by summing the clipped density of its children
            hex_raw_density = 0
            hex_unclipped_density = 0
            for c in children:
                if c in hex_density:

                    hex_raw_density += hex_density[c]['clipped']
                    hex_unclipped_density += hex_density[c]['actual']
            hex_density[h] = dict(clipped=hex_raw_density, actual=hex_unclipped_density, unclipped=hex_raw_density)

        # now that we have unclipped densities of each occupied hex at this resolution, iterate through all occupied
        # hexs again and apply clipping by looking at neighbors:

        for h in occupied_hexs:
            #neigbhors = h3.hex_range(h, 1)
            #neigbhors = h3.h3_to_children(h3.h3_to_parent(h, resolution - 1), resolution)
            clip = occupied_neighbors(h, density_res_tgt, density_res_max, N, hex_density, method='neighbors')

            hex_density[h]['clipped'] = min(hex_density[h]['clipped'], clip)
            hex_density[h]['limit'] = clip
        occupied_higher_res = list(occupied_hexs)

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

    interactive_hspots = 0

    with open(f'hex_occupancy_R{R}_N{N}_tgt{density_tgt}_max{density_max}.csv', 'w', newline='') as csvfile:
        hex_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hex_writer.writerow(['hex', 'resolution', 'density_clipped', 'density_actual', 'density_limit'])
        for k in hex_density:
            hex_writer.writerow([k, h3.h3_get_resolution(k), hex_density[k]['clipped'], hex_density[k]['actual'], hex_density[k]['limit']])

    with open(f'hotspot_RewardScale_R{R}_N{N}_tgt{density_tgt}_max{density_max}.csv', 'w', newline='') as csvfile:
        hspot_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hspot_writer.writerow(['address', 'name', 'city', 'state', 'reward_scale'])
        # iterate through all interactive hotspots and evaluate probability of targeting. this will be outputted to CSV
        for hspot in hotspots:
            # start at top level and iterate through determining odds of selection
            if not is_interactive(hspot):
                continue
            interactive_hspots += 1
            scale = 1
            probability = 1
            #for res in range(1, R+1):
            for res in range(R, 0, -1):
                hex = h3.h3_to_parent(hspot['location'], res)
                scale_orig = scale
                scale *= hex_density[hex]['clipped'] / hex_density[hex]['unclipped']
                if hspot['name'] == 'daring-carmine-penguin':
                    print(f"{hex} h3res:{res} has density clipped/unclipped of {hex_density[hex]['clipped']:3d}/{hex_density[hex]['unclipped']:3d}, scale reduced: {scale_orig:.3f} to {scale:.3f}")
                sibling_total = hex_density[hex]['clipped']
                sibling_unclipped = hex_density[hex]['actual']

            hspot_writer.writerow([hspot['address'], hspot['name'], hspot['geocode']['short_city'], hspot['geocode']['short_state'], f"{scale:.5f}"])
            # print(f"hotspot {hspot['name']:30} has {sibling_unclipped} hotspots in res8 cell, probability {probability*100:.8f}%")

        print(f"total of {interactive_hspots} interactive hotspots")




def odds_by_city(file='hotspot_RewardScale_R8_N2_tgt1_max4.csv'):
    hspots = 0
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        city_dict = dict()
        all_probs = []
        spamreader.__next__()
        for row in spamreader:
            city = f"{row[2]}, {row[3]}"
            if 'Huron' in city:
                print(city)
            city_dict.setdefault(city, [])
            city_dict[city].append(float(row[-1]))
            all_probs.append(float(row[-1]))
            hspots += 1
        axticks = []
        mymax = 0
        all_probs_norm = np.array(all_probs) * hspots
        print(f"num hotspots {hspots}")
        print(f"{np.sum(all_probs_norm>1.0)} hotspots will see increased targeting up to {np.max(all_probs_norm):.3f} times ")
        print(f"{np.sum(all_probs_norm<1.0)} hotspots will see decreased targeting up to {np.min(all_probs_norm):.3f} times ")
        city_keys = list(city_dict.keys())
        city_avg = []
        for x in city_keys:

            city_dict[x] = np.array(city_dict[x])
            city_avg.append(np.mean(city_dict[x]))

        sortidx = np.argsort(city_avg)
        data = []
        for idx in sortidx:

            if len(city_dict[city_keys[idx]]) >= 45 or (len(city_dict[city_keys[idx]]) >= 12 and (np.mean(city_dict[city_keys[idx]]) <= 0.5)):
                mymax = max(mymax, np.max(city_dict[city_keys[idx]]))
                axticks.append(f"{city_keys[idx]} [{len(city_dict[city_keys[idx]]):3d}]")
                data.append(city_dict[city_keys[idx]])

        axticks.append("All Hotspots")
        data.append(all_probs)

        plt.violinplot(
            data,
            widths=1.5,
            points=100,
            bw_method=.05,
            showmeans=True,
            vert=False
        )
        plt.yticks(range(1, 1+len(axticks)), axticks)
        plt.ylim([0, len(axticks)+1])
        plt.xlim([0, mymax * 1.02])
        plt.xlabel("normalized targeting probability (vs 1/|hotspots|)")
        plt.title(f"violin plot of Reward Scale by city\n{file}")
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
    sample_neighbor(
        hotspots=hs,
        density_tgt=1,
        density_max=4,
        R=8,
        N=2
    )

if __name__ == '__main__':
    #main()
    odds_by_city()