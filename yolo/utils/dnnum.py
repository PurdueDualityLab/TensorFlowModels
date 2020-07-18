if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.description = "Number the blocks in a DarkNet config file"
    parser.add_argument('filename', help='name of the config file')
    args = parser.parse_args()

    filename = args.filename
    with open(filename) as file:
        i = 0
        for line in file:
            if line.startswith('[') and line != '[net]\n':
                print(f"{i:4d}|{line}", end='')
                i += 1
            else:
                print(f"    |{line}", end='')
