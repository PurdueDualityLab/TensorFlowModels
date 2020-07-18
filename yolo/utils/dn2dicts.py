if __name__ == '__main__':
    import argparse
    import json
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.description = "Convert a DarkNet config file into a Python AST limited file in a dictionary format"
    parser.add_argument('config', help='name of the config file')
    parser.add_argument('dictsfile', help='name of the Python AST limited file')
    args = parser.parse_args()

    config = args.config
    dictsfile = args.dictsfile
    output = []
    mydict = None
    with open(config) as configfile:
        i = 0
        for line in configfile:
            if line.startswith('[') and line != '[net]\n':
                mydict = {}
                mydict['_type'] = line.strip('[] \n')
                output.append(mydict)
            elif mydict is not None:
                line, *_ = line.strip().split('#', 1)
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    mydict[k] = v

    with open(dictsfile, 'w') as dictsfilew:
        pprint(output, dictsfilew)
