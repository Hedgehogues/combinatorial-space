from tqdm import tqdm

def convert(imgf, labelf, outf, n, size):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in tqdm(range(n), desc=outf):
        image = [ord(l.read(1))]
        for j in range(size[0]*size[1]):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()
