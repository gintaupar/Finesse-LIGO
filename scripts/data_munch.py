#!/usr/bin/env python3

import gwpy.timeseries as ts
from munch import Munch
import yaml
from joblib import Memory


if __name__ == "__main__":
    import argparse
    import pickle

    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', action='store_true',
                   help='verbose nds2 fetching')
    p.add_argument('channel_file',
                   help='path to channel spec yaml file',
                   type=str)
    p.add_argument('output',
                   help='path to output file')

    args = p.parse_args()
    VERBOSE = args.verbose
    CHANNEL_FILE = args.channel_file

    @Memory('_cache').cache
    def get_data(channels, start, end):
        tfd = ts.TimeSeriesDict.get(
            channels,
            start=start, end=end,
            verbose=VERBOSE
        )
        return tfd

    def munch_yaml(chan_file=CHANNEL_FILE):
        with open(chan_file, 'r') as f:
            yamldata = yaml.safe_load(f)

        chans = [x['Name'] for x in yamldata['Channels']]
        spans = yamldata['Spans']

        data = Munch()
        for span in spans:
            s, e = span
            tfd = get_data(chans, s, e)
            data[s] = {}
            data[s]['data'] = tfd
            data[s]['span'] = span
            data[s]['chans'] = chans
            data[s]['metadata'] = yamldata['Channels']

        return data

    x = munch_yaml()
    with open(args.output, 'wb') as fout:
        pickle.dump(x, fout)
