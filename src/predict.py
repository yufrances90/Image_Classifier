from utils.io import IO

def main():

    in_arg = IO.get_input_args(train=False)

    device = IO.get_device(in_arg.gpu)

    category_names = IO.get_label_mapping(in_arg.category_names)

    print(category_names)

if __name__ == "__main__":
    main()