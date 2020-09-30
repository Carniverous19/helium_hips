import h3
import json
from utils import load_hotspots
import csv

class Interactive:
    def __init__(self):
        hotspots = load_hotspots()
        self.h_by_addr = dict()
        for h in hotspots:
            self.h_by_addr[h['address']] = h
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

        self.interactives = interactives

    def is_interactive(self, hotspot_address):
        return hotspot_address in self.interactives and self.h_by_addr[hotspot_address]['location']


class RewardScale:
    def __init__(self, chain_vars):

        I = Interactive()


        self.hotspots = []
        self.h_by_addr = dict()
        for h in load_hotspots():
            if I.is_interactive(h['address']):
                self.hotspots.append(h)
                self.h_by_addr[h['address']] = h



        self.chain_vars = chain_vars



    def __clip_hex__(self, hex, hex_densities):
        """

        :param hex: hex string to evaluate
        :param hex_densities: dictionary of all hex densities, atleast at this resolution
        :return: this hex density clipped based on neigbhors
        """
        res = h3.h3_get_resolution(hex)
        res_key = str(res)
        neighbors = h3.hex_range(hex, 1)
        at_tgt_count = 0
        for n in neighbors:
            if hex_densities.get(n, 0) >= self.chain_vars['res_vars'][res_key]['density_tgt']:
                at_tgt_count += 1

        clip = min(
            self.chain_vars['res_vars'][res_key]['density_max'],
            self.chain_vars['res_vars'][res_key]['density_tgt'] * max(1, (at_tgt_count - self.chain_vars['res_vars'][res_key]['N'] + 1))
        )
        val = min(clip, hex_densities[hex])
        if val != hex_densities[hex]:
            print(f'clipped {hex} at res {res} from {hex_densities[hex]} to {val}')
        return val

    def get_hex_densities(self):

        # first build densities at target resolution R
        target_hex_unclipped = dict()
        for h in self.hotspots:
            hex = h3.h3_to_parent(h['location'], self.chain_vars['R'])
            target_hex_unclipped.setdefault(hex, 0)
            target_hex_unclipped[hex] += 1

        hex_densities = dict()
        # clip targets so we have valid children in hex densities to begin ascending through list
        for h in target_hex_unclipped:
            hex_densities[h] = self.__clip_hex__(h, target_hex_unclipped)

        # now we initialized hex_densities with appropriately clipped target hexs go from resolution R-1 to 0
        occupied_children = set(list(hex_densities.keys()))
        for res in range(self.chain_vars['R']-1, -1, -1):

            # iterate through children getting uncipped density for each hex at this res
            occupied_hexs = set([])
            for child_hex in occupied_children:
                hex = h3.h3_to_parent(child_hex, res)
                occupied_hexs.add(hex)
                hex_densities.setdefault(hex, 0)
                hex_densities[hex] += hex_densities[child_hex]

            print(f"found {len(occupied_hexs):4d} occupied hexs at resolution {res}")
            # clip hex's at this res as appropriate
            for hex in occupied_hexs:
                hex_densities[hex] = self.__clip_hex__(hex, hex_densities)

            occupied_children = occupied_hexs

        return hex_densities, target_hex_unclipped

    def get_reward_scale(self, hex_densities, target_hex_unclipped, whitelist_hexs, normalize=False):
        """

        :param hex_densities: dict of densiteis of each occupied hex at all levels
        :param whitelist_hexs: dict of densities of whitelisted hexs
        :param target_hex_unclipped: dict of res R hex as keys and raw count of interactive hexs as values
            this could be regenerated pretty easily if desired (O(|hotspots|)) if is_interactive is O(1) and fast
        :return:
        """
        reward_scales = dict()
        whitelist_density = 0
        for whex in whitelist_hexs:
            whitelist_density += hex_densities.get(whex, 0)
        for h in self.hotspots:
            print(f"analyzing {h['name']}")
            # initialize scale by uniformly selecting among hotsots in target res parent
            hspot_hex = h3.h3_to_parent(h['location'], self.chain_vars['R'])
            scale = hex_densities[hspot_hex] / target_hex_unclipped[h3.h3_to_parent(h['location'], self.chain_vars['R'])]
            print(f"\t initial scale {scale:.5f}")
            for parent_res in range(self.chain_vars['R']-1, -1, -1):
                if hspot_hex in whitelist_hexs:
                    break
                parent = h3.h3_to_parent(h['location'], parent_res)
                children_sum = 0

                for child in h3.h3_to_children(parent, parent_res + 1):
                    children_sum += hex_densities.get(child, 0)

                scale *= hex_densities[parent]/children_sum
                print(f"\tat res {parent_res+1} scale at: {scale:.5f}")
                hspot_hex = parent

            if hspot_hex not in whitelist_hexs:
                scale = 0
            normalized_scale = scale * len(self.hotspots)
            print(f"\tnormalized scale: {normalized_scale:.5f}")
            reward_scales[h['address']] = scale

        if normalize:
            # will set mean of all scales to 1 for ease of understanding
            scale_sum = 0
            for v in reward_scales.values():
                scale_sum += v
            scale_avg = scale_sum / len(reward_scales)
            for k in reward_scales.keys():
                reward_scales[k] /= scale_avg

        return reward_scales







def main():
    with open('chain_vars.json', 'r') as fd:
        chain_vars = json.load(fd)
    print(chain_vars)

    # for now set all level 0 hex with a hotspot as whitelist
    whitelist_hexs = set()
    for h in load_hotspots():
        if h['location']:
            whitelist_hexs.add(h3.h3_to_parent(h['location'], 0))

    RS = RewardScale(chain_vars)
    hex_densities, target_hex_unclipped = RS.get_hex_densities()

    hotspot_scales = RS.get_reward_scale(hex_densities, target_hex_unclipped, whitelist_hexs=whitelist_hexs, normalize=True)
    total_scale = 0
    for v in hotspot_scales.values():
        total_scale += v


    with open(f'hotspot_RewardScale_R{chain_vars["R"]}.csv', 'w', newline='') as csvfile:
        hex_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hex_writer.writerow(['address', 'reward_scale'])
        for h in hotspot_scales:
            hex_writer.writerow([h, hotspot_scales[h]])
    print(f"total scale at {total_scale}, with {len(hotspot_scales)} hotspots")
    print(f"average scale = {total_scale/len(hotspot_scales):.5f}")


if __name__ == '__main__':
    main()
