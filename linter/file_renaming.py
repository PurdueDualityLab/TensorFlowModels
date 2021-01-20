# TODO: Finish this file
import pasta

for file in files:
    with open(file) as src_file:
        src = src_file.read()
    ast = pasta.parse(src)
    import code
    code.interact(local=vars())
    src = pasta.dump(ast)
    with open(file, 'w') as src_file:
        src_file.write(src)
