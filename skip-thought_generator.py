from skipthoughts import skipthoughts
import h5py

def main():
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)

    caption_file = "captions.txt"
    training_image_file = "training_images.txt"

    captions = []
    with open(caption_file) as f:
        line_list = f.read().split("\n")
        f1 = open(training_image_file, "w")
        for line in line_list:
            img = line.split("\t")[0]
            cap = line.split("\t")[1]
            if len(cap) > 0:
                captions.append(cap)
                f1.write(img + "\n")
        f1.close()

    caption_vectors = encoder.encode(captions)

    h = h5py.File("caption_vectors.hdf5", "w")
    h.create_dataset("vectors", data=caption_vectors)		
    h.close()

if __name__ == '__main__':
	main()
