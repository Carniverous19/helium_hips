import csv
from utils import load_hotspots, haversine_km
import random
import folium
import numpy as np
import math
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import h3


class Witnesses:
    """
    Build real world witness lookup
    """
    def load_wit_file(self, file):
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            count = 0
            results = []
            for row in reader:
                tx = row[0].strip()
                wit = row[1].strip()
                if tx not in self.witnesses:
                    self.witnesses[tx] = dict()
                if wit in self.witnesses[tx]:
                    continue
                # ignore hotspots that are offline
                if self.h_by_addr[tx]['status']['online'] != 'online':
                    continue
                if self.h_by_addr[wit]['status']['online'] != 'online':
                    continue
                dist = haversine_km(
                    self.h_by_addr[tx]['lat'],
                    self.h_by_addr[tx]['lng'],
                    self.h_by_addr[wit]['lat'],
                    self.h_by_addr[wit]['lng']
                )
                if 0.3 < dist < 50:
                    self.witnesses[tx][wit] = dist

    def __init__(self):
        hs = load_hotspots()
        self.h_by_addr = dict()
        for h in hs:
            self.h_by_addr[h['address']] = h

        # key = hotspot address
        # value = dict with keys of witness address, values = distance
        # only witnesses > 300m will appear
        self.witnesses = dict()

        # try:
        #     self.load_wit_file('witnesses.csv')
        # except FileNotFoundError as e:
        #     pass
        try:
            self.load_wit_file('witnesses2.csv')
        except FileNotFoundError as e:
            pass

    def get_witnesses(self, txaddr, return_probs=False):
        if txaddr not in self.witnesses:
            return dict()
        if len(self.witnesses[txaddr]) > 250:
            return dict()
        if not return_probs:
            return list(self.witnesses[txaddr].keys())

        # rough probability of success estimate
        mywits = dict()
        for w in self.witnesses[txaddr].keys():
            prob = .75
            if self.witnesses[txaddr][w] > .4:
                prob = min(prob, math.pow(.4/self.witnesses[txaddr][w], 1/2))

            mywits[w] = prob
        return mywits



    def simulate_transmit(self, txaddr, num_txs=1, normalize=False, scale=1.0):
        """

        :param txaddr: hotsot address

        :param num_txs: number of transmits to simulate
        :param normalize: if True divide rewards by number of transmissions so its normalized per tx
        :return: dictionary of witnesses and transmitter with rewards for each
        """
        rewards = {txaddr: 0}
        wit_dict = self.get_witnesses(txaddr, return_probs=True)
        if len(wit_dict) == 0:
            return rewards
        print(f"\thotspot has {len(wit_dict)} witnesses")
        for i in range(0, num_txs):
            witnesses = []

            # find witnesses for this transmission
            for k in wit_dict.keys():
                if random.random() < wit_dict[k]:
                    witnesses.append(k)
            if not witnesses:
                continue

            # calculate rx and tx rewards
            tx_reward = min(4, len(witnesses)) / 4 * 0.2
            rx_reward = 1
            if len(witnesses) > 4:
                tx_transfer = (1-.8**(len(witnesses)-4))
                tx_reward += tx_transfer
                rx_reward = (4 - tx_transfer) / len(witnesses)

            #print(f"\t{len(witnesses)}/{len(wit_dict.keys())} witnesses. scale {scale:.2f}: txrw:{tx_reward:.2f}, rxrw:{rx_reward:.2f}")
            # give rewards to hotspots
            rewards[txaddr] += tx_reward
            for k in witnesses:
                rewards.setdefault(k, 0)
                rewards[k] += rx_reward

        if normalize:
            for k in rewards:
                rewards[k] /= num_txs
        for k in rewards:
            rewards[k] *= scale
        return rewards

    def output_witness_edges(self, outfile='my_witnesses.csv'):
        txs = set([])
        with open(outfile, 'w', newline='') as csvfile:
            hex_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            hex_writer.writerow(['tx_addr', 'rx_addr', 'witness_prob', 'dist_km'])
            for h in self.witnesses:
                txs.add(h)

            for h in txs:
                wits = self.get_witnesses(h, return_probs=True)
                for w in wits.keys():
                    hex_writer.writerow([h, w, round(wits[w], 5), round(self.witnesses[h][w], 2)])



def get_hotspot_scales( fn=None):

    with open(fn, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        reader.__next__()
        count = 0
        results = []
        for row in reader:
            count += 1
            results.append(dict(addr=row[0], odds=float(row[-1])))
    return results


def sigmoid(x):
    return math.exp(x)/(1 + math.exp(x))

def map_hotspot_rewards(outputfile='expected_rewards.html', bbox=None):
    """

    :param outputfile:
    :param bbox: lat upper left, long upper left, lat upper right, long upper right
    :return:
    """

    hs = []
    h_by_addr = dict()
    for h in load_hotspots():

        hs.append(h)
        h_by_addr[h['address']] = h


    lat = (bbox[0] + bbox[2])/2
    lng = (bbox[1] + bbox[3])/2

    Wits = Witnesses()
    #Wits.output_witness_edges('sim_witness_edges.csv')
    #hspots = get_hotspot_scales(fn='hotspot_RewardScale_R9.csv')

    # ignore hex scaling just simulate beaconing
    hspots = []
    for h in hs:
        hspots.append(dict(addr=h['address'], odds=1.0))
    rewards = dict()


    # simulate transmissions
    for h in hspots:
        haddr = h['addr']
        #print(f"simulated rewards for {haddr}")
        rew = Wits.simulate_transmit(txaddr=haddr, num_txs=500, scale=h['odds'], normalize=True)
        if h_by_addr[haddr]['name'] == 'recumbent-magenta-aphid':
            for h in rew.keys():
                print(f"  neighor: {h}, reward={rew[h]:.3f} w/ scale ")
        for r in rew:
            rewards.setdefault(r, 0)
            rewards[r] += rew[r]
            if r in h_by_addr and h_by_addr[r]['name'] == 'faint-pecan-trout':
                print(f"faint earned: {rew[r]:.4f}, at: {rewards[r]}")

    rw_sum = 0
    rw_max = 0
    max_h = None
    cnt = 0
    for h in rewards.keys():

        #rewards[h] = math.sqrt(rewards[h])
        x = rewards[h]
        if x == 0:
            continue
        #print(f"{h} rw: {x:.3f}")
        rw_max = max(rw_max, x)
        if rw_max == x:
            max_h = h
        rw_sum += x
        cnt += 1
    rw_avg = rw_sum/cnt
    print(f"max earner {h_by_addr[max_h]['name']}, earned {rw_max:.3f} with {len(Wits.get_witnesses(max_h))} witnesses")
    print(f"max at {h_by_addr[max_h]['geocode']}")



    plt.figure()
    rw = np.array(list(rewards.values()))
    pct = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    qs = np.quantile(rw[rw > 0], pct)
    for i in range(0, len(qs)):
        print(f"{pct[i]*100}% cutoff at {qs[i]:.3f}")

    plt.hist(rw[rw > 0], bins=50)
    plt.xlabel("simulated reward units")
    plt.ylabel("hotspot count")
    plt.title("Reward distribution for HIP15")
    plt.grid()
    plt.show()

    make_map = True
    if make_map:
        my_map = folium.Map(location=[lat, lng], zoom_start=6)

        vals = [x['odds'] for x in hspots]

        cnorm = mpl.colors.TwoSlopeNorm(vcenter=5, vmin=0, vmax=9)
        scalemap = cm.ScalarMappable(norm=cnorm, cmap='RdYlGn')

        #
        # cnorm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=(np.median(rw)), vmax=(rw_max))
        # earnmap = cm.ScalarMappable(norm=cnorm, cmap='cool')

    scale_dict = dict()
    for h in hspots:
        scale_dict[h['addr']] = h['odds']

    idc = np.argsort(list(rewards.values()))
    hs = np.array(list(rewards.keys()))
    with open('simulated_rewards_bcn.csv', 'w', newline='') as csvfile:
        reward_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        reward_writer.writerow(['address', 'h3', 'lat', 'lng', 'tx_rw_scale', 'E_reward'])
        for h in hs[idc]:
            if h not in h_by_addr:
                continue
            if rewards[h] == 0:
                continue
            hlat = h_by_addr[h]['lat'] #+ (random.random()-0.5)*.0000
            hlng = h_by_addr[h]['lng'] #+ (random.random()-0.5)*.0000

            if not (bbox[2] < hlat < bbox[0] and bbox[1] < hlng < bbox[3]):
                continue
            if h not in scale_dict:
                continue

            reward_writer.writerow([
                h,
                h_by_addr[h]['location'],
                f"{h_by_addr[h]['lat']:.5f}",
                f"{h_by_addr[h]['lng']:.5f}",
                f"{scale_dict[h]:.4f}",
                f"{rewards[h]:.3f}"
            ])

            if make_map:
                idx = np.searchsorted(qs,[rewards[h]])[0]
                color_body = "#" + ''.join([f"{x:02x}" for x in scalemap.to_rgba(idx, bytes=True)[:3]])
                #color_body = "#" + ''.join([f"{x:02x}" for x in earnmap.to_rgba(rewards[h], bytes=True)[:3]])
                if idx == 0:
                    qstr = f"<{pct[0]*100:.0f}"
                elif idx == len(qs):
                    qstr = f">{pct[-1] * 100:.0f}"
                else:
                    qstr = f"{pct[idx-1] * 100:.0f}-{pct[idx] * 100:.0f}"
                folium.CircleMarker(
                    (hlat, hlng), color=color_body, fill_color=color_body,
                    popup=f"<nobr>{h_by_addr[h]['name']}</nobr><br>rew. units:{(rewards[h]):.2f}<br><nobr>rew. pct:{qstr}%</nobr><br>#wits:{len(Wits.get_witnesses(h))}",
                    fill=True, fill_opacity=0.55,
                    number_of_sides=8, radius=11, opacity=1, weight=3
                ).add_to(my_map)
    if make_map:
        my_map.save(outputfile)

def map_reward_file(filename='real_rewards.csv', bbox=[0, 0, 0, 0]):


    hs = []
    h_by_addr = dict()
    for h in load_hotspots():

        hs.append(h)
        h_by_addr[h['address']] = h

    lat = bbox[0]/2 + bbox[2]/2
    lng = bbox[1]/2 + bbox[3]/2
    my_map = folium.Map(location=[lat, lng], zoom_start=6)
    cnorm = mpl.colors.TwoSlopeNorm(vcenter=5, vmin=0, vmax=9)
    scalemap = cm.ScalarMappable(norm=cnorm, cmap='RdYlGn')

    with open(filename, 'r', newline='') as csvfile:
        reward_reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        reward_reader.__next__()
        rewards = []

        for row in reward_reader:
            h = row[0]
            hlat = h_by_addr[h]['lat']
            hlng = h_by_addr[h]['lng']
            if h not in h_by_addr:
                continue
            if not (bbox[2] < hlat < bbox[0] and bbox[1] < hlng < bbox[3]):
                continue
            rewards.append([row[0], float(row[-1])])

    rw = np.array([r[-1] for r in rewards])
    pct = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    qs = np.quantile(rw[rw > 0], pct)
    for i in range(0, len(qs)):
        print(f"{pct[i]*100}% cutoff at {qs[i]:.3f}")

    for r in rewards:
        h = r[0]
        hlat = h_by_addr[h]['lat'] #+ (random.random()-0.5)*.0000
        hlng = h_by_addr[h]['lng'] #+ (random.random()-0.5)*.0000

        if not (bbox[2] < hlat < bbox[0] and bbox[1] < hlng < bbox[3]):
            continue



        idx = np.searchsorted(qs,[r[-1]])[0]
        color_body = "#" + ''.join([f"{x:02x}" for x in scalemap.to_rgba(idx, bytes=True)[:3]])
        #color_body = "#" + ''.join([f"{x:02x}" for x in earnmap.to_rgba(rewards[h], bytes=True)[:3]])
        if idx == 0:
            qstr = f"<{pct[0]*100:.0f}"
        elif idx == len(qs):
            qstr = f">{pct[-1] * 100:.0f}"
        else:
            qstr = f"{pct[idx-1] * 100:.0f}-{pct[idx] * 100:.0f}"
        folium.CircleMarker(
            (hlat, hlng), color=color_body, fill_color=color_body,
            popup=f"<nobr>{h_by_addr[h]['name']}</nobr><br>rew. units:{r[-1]:.2f}<br><nobr>rew. pct:{qstr}%</nobr>",
            fill=True, fill_opacity=0.55,
            number_of_sides=8, radius=11, opacity=1, weight=3
        ).add_to(my_map)

    my_map.save('para1_realrewards.html')

if __name__ == '__main__':

    USA_bbox = [48.922499, -124.872527, 24.806681, -66.896522]
    map_hotspot_rewards(bbox=USA_bbox)
    #map_reward_file(bbox=USA_bbox)


