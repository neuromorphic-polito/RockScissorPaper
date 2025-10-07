from accumulate_to_image import *

#Uses categories {paper: 0, rock: 2, scissors: 1, background: 3} for 
CATEGORIES = {
    0: "PAPER",
    1: "SCISSORS",
    2: "ROCK",
    3: "BACKGROUND"
}

def category(prediction):
    print(CATEGORIES.get(prediction, "UNKNOWN"))



root_sd = "test_set_scf/davis/"

#40k,  /home/cass/work/SCF/test_set_scf/davis/davis_carta.aedat4
ls_img = aedat4_to_images(filename=root_sd+"davis_carta.aedat4",
                            acc_value=40000)


#save_images_to_folder(ls_img, labels=[3]*len(ls_img), out_dir="./frames/bg")
