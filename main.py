from model import MtoE1, MtoE2, MtoE3

def main():
    model = MtoE3.load()

    model.run_training(1)

    model.evaluate([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

if __name__=="__main__":
    main()