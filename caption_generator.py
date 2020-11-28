import pandas as pd

DATASET_SIZE = 202599
TRAINING_SIZE = 10544
DATA_SIZE = 50000

class CaptionGenerator:
    def __init__(self):
        self.attributes = [ "5'O clock shadow", "arched", "attractive", "bags under eyes", "bald",
        "bangs", "big lips", "a big nose", "black", "blond", "blurry", "brown", "bushy", "chubby",
        "a double chin", "eyeglasses", "goatee", "gray", "heavy makeup", "high cheekbones", "male",
        "slightly open mouth", "mustache", "narrow eyes", "no beard", "an oval face", "pale skin", 
        "a pointy nose", "Receding hairline", "rosy cheeks", "Sideburns", "smiling", "straight", "wavy",
        "earrings", "a hat", "lipstick", "a necklace", "a necktie", "young" ]

    def get_caption (self, img_idx):
        self.img = img_idx

        if (img_idx[20]):
            self.noun = "man"
            self.pronoun = "He"
            self.pronoun2 = "His"
        else:
            self.noun = "woman"
            self.pronoun = "She"
            self.pronoun2 = "Her"
        
        full_caption = ""
        full_caption += self.broad_features()
        full_caption += self.hair_attributes()
        full_caption += self.eye_attributes()
        full_caption += self.accessories()
        full_caption += self.mouth_description()
        return full_caption

    def create_conjunction(self, list_attr):
        size = len(list_attr)
        if (size == 1):
            return self.attributes[list_attr[0]] + " "

        ans = ""
        for i in range(size-1):
            ans += self.attributes[list_attr[i]]
            ans += ", "

        ans = ans[:-2]
        ans = ans + " and " + self.attributes[list_attr[size-1]]
        return ans 

    def broad_features(self):
        caption = "The "
        if (self.img[2] or self.img[10] or self.img[13] or self.img[39]):
            list_attr = []
            if (self.img[2]):
                list_attr.append(2)
            if (self.img[10]):
                list_attr.append(10)
            if (self.img[13]):
                list_attr.append(13)
            if (self.img[39]):
                list_attr.append(39)
            caption += self.create_conjunction(list_attr)
        caption += " " + self.noun

        if (self.img[6] or self.img[7] or self.img[14] or self.img[19] or self.img[25] or self.img[26]
        or self.img[27] or self.img[29]):
            list_attr = []
            if (self.img[6]):
                list_attr.append(6)
            if (self.img[7]):
                list_attr.append(7)
            if (self.img[14]):
                list_attr.append(14)
            if (self.img[19]):
                list_attr.append(19)
            if (self.img[25]):
                list_attr.append(25)
            if (self.img[26]):
                list_attr.append(26) 
            if (self.img[27]):
                list_attr.append(27)
            if (self.img[29]):
                list_attr.append(29)
            caption += " has " + self.create_conjunction(list_attr) + "."

        return caption

    def hair_attributes(self):
        caption = ""
        if (self.img[4]):
            caption += " " + self.pronoun + " is " + self.attributes[4]
            if (self.img[24]):
                caption += " with " + self.attributes[24]
            caption += ". "
            return caption
        
        if (self.img[32] or self.img[33]):
            hairstyle = ""
            if (self.img[32]):
                hairstyle = self.attributes[32]
            else:
                hairstyle = self.attributes[33]
            caption += " " + self.pronoun + " has " + hairstyle + " hair"
            if (self.img[8] or self.img[9] or self.img[11]):
                colour = ""
                if (self.img[8]):
                    colour = self.attributes[8]
                elif (self.img[9]):
                    colour = self.attributes[9]
                else:
                    colour = self.attributes[11]
                caption += " which are " + colour + " in colour."
            caption += "."

        elif (self.img[8] or self.img[9] or self.img[11]):
                colour = ""
                if (self.img[8]):
                    colour = self.attributes[8]
                elif (self.img[9]):
                    colour = self.attributes[9]
                else:
                    colour = self.attributes[11]
                caption += " " + self.pronoun + " has " + colour + " coloured hair."

        if (self.img[28] or self.img[30]):
            list_attr = []
            if (self.img[28]):
                list_attr.append(28)
            if (self.img[30]):
                list_attr.append(30)
            caption += " " + self.create_conjunction(list_attr) + "is/are evident."
        
        if (self.img[5] or self.img[24]):
            caption += " " + self.pronoun + " has "
            list_attr = []
            if (self.img[5]):
                list_attr.append(5)
            if (self.img[24]):
                list_attr.append(24)
            caption += self.create_conjunction(list_attr) + "."
        
        if (self.img[0] or self.img[16] or self.img[22]):
            caption += " " + self.pronoun + " exhibits a "
            list_attr = []
            if (self.img[0]):
                list_attr.append(0)
            if (self.img[16]):
                list_attr.append(16)
            if (self.img[22]):
                list_attr.append(22)
            caption += self.create_conjunction(list_attr) + "."

        return caption

    def eye_attributes(self):
        caption = ""
        if (self.img[3] or self.img[23]):
            list_attr = []
            if (self.img[3]):
                list_attr.append(3)
            if (self.img[23]):
                list_attr.append(23)
            caption += " " + self.pronoun + " has " + self.create_conjunction(list_attr)

            if (self.img[15]):
                caption += " and wears " + self.attributes[15]
            caption += "."

        elif (self.img[15]):
            caption += " " + self.pronoun + " wears " + self.attributes[15] + "."

        if (self.img[1] or self.img[12]):
            list_attr = []
            if (self.img[1]):
                list_attr.append(1)
            if (self.img[12]):
                list_attr.append(12)
            caption += " " + self.pronoun2 + " eyebrows are " + self.create_conjunction(list_attr) + "."
        
        return caption

    def accessories(self):
        caption = ""
        if (self.img[18] or self.img[34] or self.img[35] or self.img[36] or self.img[37] or self.img[38]):
            list_attr = []
            if (self.img[18]):
                list_attr.append(18)
            if (self.img[34]):
                list_attr.append(34)
            if (self.img[35]):
                list_attr.append(35)
            if (self.img[36]):
                list_attr.append(36)
            if (self.img[37]):
                list_attr.append(37)
            if (self.img[38]):
                list_attr.append(38)
            caption += " " + self.pronoun + " is wearing " + self.create_conjunction(list_attr) + "."
        return caption

    def mouth_description(self):
        caption = ""
        if (self.img[21]):
            caption += " " + self.pronoun + " has a " + self.attributes[21] + "."
        if (self.img[31]):
            caption += " The " + self.noun + " is " + self.attributes[31] + "."
        return caption

def main():
    cg = CaptionGenerator()
    data = pd.read_csv("list_attr_celeba.txt", sep=" ")
    captions_file = open("captions.txt", "w")
    for i in range(DATA_SIZE):
        image = data.iloc[i]
        image_name = image[0]
        image = image[1:]
        cap = cg.get_caption(image)
        captions_file.write(image_name + " " + cap + "\n")
    captions_file.close()

if __name__ == '__main__':
	main()
