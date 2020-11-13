import h3
import json
from utils import load_hotspots, haversine_km
import csv

class Interactive:
    """
    Uses LUT of two separate dumps of witness lists per @evan's code to get addresses only witness list:
    CSV = maps:fold(fun(K, V, A) -> W = blockchain_ledger_gateway_v2:witnesses(V), [maps:fold(fun(K2, _, A2) ->  [io_lib:format("~s, ~s~n", [libp2p_crypto:bin_to_b58(K), libp2p_crypto:bin_to_b58(K2)]) | A2] end, [], W)| A] end, [], blockchain_ledger_v1:active_gateways(blockchain:ledger(blockchain_worker:blockchain()))), file:write_file("/var/witnesses.csv", CSV).

    File Format will be:
    11pu4B7uGTiXATZunczzYk4iefpkt4cJf7zm9q4ZvTxr63WvmUP, 112mgSjGUZCBit2sv3Vv2FuLwaTVnMJHg3XN8H1xjAqdyc5t9hzZ
    11pu4B7uGTiXATZunczzYk4iefpkt4cJf7zm9q4ZvTxr63WvmUP, 112mEVXHvkYTpHJcE5SGQo42bKDFdKh2fiTexjuMLnmr3XdREXi6
    11pu4B7uGTiXATZunczzYk4iefpkt4cJf7zm9q4ZvTxr63WvmUP, 112kYX5HpnuoD7WPjDRYPV2VCmZvwtSEvCuyi2XZLoeqsVvhiF5h
    11pu4B7uGTiXATZunczzYk4iefpkt4cJf7zm9q4ZvTxr63WvmUP, 112euXBKmLzUAfyi7FaYRxRpcH5RmfPKprV3qEyHCTt8nqwyVFYo
    11pu4B7uGTiXATZunczzYk4iefpkt4cJf7zm9q4ZvTxr63WvmUP, 112eNuzPYSeeo3tqNDidr2gPysz7QtLkePkY5Yn1V7ddNPUDN6p5
    ...
    """
    def __init__(self):
        hotspots = load_hotspots()
        self.h_by_addr = dict()
        for h in hotspots:
            self.h_by_addr[h['address']] = h
        interactives = set([])
        found_witness = False
        try:
            with open('witnesses.csv', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                count = 0
                results = []
                for row in reader:

                    dist = haversine_km(
                        self.h_by_addr[row[0].strip()]['lat'],
                        self.h_by_addr[row[0].strip()]['lng'],
                        self.h_by_addr[row[1].strip()]['lat'],
                        self.h_by_addr[row[1].strip()]['lng']
                    )
                    if dist < 0.3:
                        continue
                    interactives.add(row[0].strip())
                    #interactives.add(row[1].strip())
                found_witness = True
        except FileNotFoundError as e:
            pass
        try:
            with open('witnesses2.csv', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                count = 0
                results = []
                for row in reader:
                    dist = haversine_km(
                        self.h_by_addr[row[0].strip()]['lat'],
                        self.h_by_addr[row[0].strip()]['lng'],
                        self.h_by_addr[row[1].strip()]['lat'],
                        self.h_by_addr[row[1].strip()]['lng']
                    )
                    if dist < 0.3:
                        continue
                    interactives.add(row[0].strip())
                    #interactives.add(row[1].strip())
                found_witness = True
        except FileNotFoundError as e:
            pass
        if not found_witness:
            print("WARNING no 'witnesses.csv' found with addresses of interactive hotsots, will assume all hotspots interactive")
            print(f"\tthis will run to understand the algorithm but will give very wrong results")
            interactives = [h['address'] for h in hotspots]
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



    def __clip_hex__(self, hex, hex_densities, return_clip=False):
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
        if return_clip:
            return val, clip
        return val

    def get_hex_densities(self):

        # first build densities at target resolution R
        target_hex_unclipped = dict()
        all_info = dict()
        for h in self.hotspots:
            hex = h3.h3_to_parent(h['location'], self.chain_vars['R'])
            target_hex_unclipped.setdefault(hex, 0)
            all_info.setdefault(hex, dict(unclipped=0, limit=-1))
            target_hex_unclipped[hex] += 1
            all_info[hex]['unclipped'] += 1

        hex_densities = dict()
        # clip targets so we have valid children in hex densities to begin ascending through list
        for h in target_hex_unclipped:
            hex_densities[h], all_info[h]['limit'] = self.__clip_hex__(h, target_hex_unclipped, return_clip=True)
        print(f"{len(self.hotspots)} interactive hotspots")
        print(f"found {len(hex_densities):4d} occupied hexs at resolution {self.chain_vars['R']}")
        # now we initialized hex_densities with appropriately clipped target hexs go from resolution R-1 to 0
        occupied_children = set(list(hex_densities.keys()))
        for res in range(self.chain_vars['R']-1, -1, -1):

            # iterate through children getting uncipped density for each hex at this res
            occupied_hexs = set([])
            for child_hex in occupied_children:
                hex = h3.h3_to_parent(child_hex, res)
                occupied_hexs.add(hex)
                hex_densities.setdefault(hex, 0)
                all_info.setdefault(hex, dict(unclipped=0, limit=-1))
                hex_densities[hex] += hex_densities[child_hex]
                all_info[hex]['unclipped'] += hex_densities[child_hex]

            print(f"found {len(occupied_hexs):4d} occupied hexs at resolution {res}")
            # clip hex's at this res as appropriate
            for hex in occupied_hexs:
                hex_densities[hex], all_info[hex]['limit'] = self.__clip_hex__(hex, hex_densities, return_clip=True)

            occupied_children = occupied_hexs

        return hex_densities, all_info

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

            # initialize scale initially set to clipped/unclipped count for target res
            hspot_hex = h3.h3_to_parent(h['location'], self.chain_vars['R'])
            scale = hex_densities[hspot_hex] / target_hex_unclipped[h3.h3_to_parent(h['location'], self.chain_vars['R'])]

            for parent_res in range(self.chain_vars['R']-1, -1, -1):
                if hspot_hex in whitelist_hexs:
                    break
                parent = h3.h3_to_parent(h['location'], parent_res)
                children_sum = 0

                for child in h3.h3_to_children(parent, parent_res + 1):
                    children_sum += hex_densities.get(child, 0)
                # multiply scale by ratio of clipped values
                scale *= hex_densities[parent]/children_sum
                hspot_hex = parent

            # if we stopped an arent at a whitelisted hex, this hex gets 0 rewards
            if hspot_hex not in whitelist_hexs:
                scale = 0

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

    # for now set all level 0 hex with a hotspot as whitelist
    whitelist_hexs = set()
    for h in load_hotspots():
        if h['location']:
            whitelist_hexs.add(h3.h3_to_parent(h['location'], 0))

    RS = RewardScale(chain_vars)
    hex_densities, all_hex_info = RS.get_hex_densities()
    with open(f'hexDensities_RewardScale_R{chain_vars["R"]}.csv', 'w', newline='') as csvfile:
        hex_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hex_writer.writerow(['hex', 'resolution', 'limit', 'clipped', 'child_sum', 'ratio'])
        for h in hex_densities:
            res = h3.h3_get_resolution(h)
            sum = all_hex_info[h]['unclipped']

            ratio = 0
            if sum:
                ratio = hex_densities[h]/sum
            hex_writer.writerow([h, res, all_hex_info[h]['limit'], hex_densities[h], sum, ratio])

    target_hex_unclipped = dict()
    for h in all_hex_info:
        target_hex_unclipped[h] = all_hex_info[h]['unclipped']
    hotspot_scales = RS.get_reward_scale(hex_densities, target_hex_unclipped, whitelist_hexs=whitelist_hexs, normalize=True)
    total_scale = 0
    for v in hotspot_scales.values():
        total_scale += v


    with open(f'hotspot_RewardScale_R{chain_vars["R"]}.csv', 'w', newline='') as csvfile:
        hex_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hex_writer.writerow(['address', 'reward_scale'])
        for h in hotspot_scales:
            hex_writer.writerow([h, hotspot_scales[h]])



if __name__ == '__main__':
    main()
