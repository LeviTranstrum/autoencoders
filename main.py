from model import MtoE

def main():
    model = MtoE.load()

    model.run_training()

    model.evaluate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

if __name__=="__main__":
    main()