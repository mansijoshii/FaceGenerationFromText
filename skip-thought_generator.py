from skipthoughts import skipthoughts
import h5py

def main():
    caption_file = "captions.txt"
    training_image_file = "train_images4.txt"

    captions = []
    with open(caption_file) as f:
        line_list = f.read().split("\n")
        line_list = line_list[7500:9000]
        f1 = open(training_image_file, "w")
        for i in range(len(line_list)):
            img = line_list[i].split("\t")[0]
            cap = line_list[i].split("\t")[1]
            if len(cap) > 0:
                captions.append(cap)
                f1.write(img + "\n")
        f1.close()

    model = skipthoughts.load_model()
    caption_vectors = skipthoughts.encode(model, captions)

    h = h5py.File("/content/drive/MyDrive/train_caption_vectors4.hdf5", "w")
    h.create_dataset("vectors", data=caption_vectors)		
    h.close()

if __name__ == '__main__':
	main()
