from accelerate import Accelerator

def main():
    accelerator = Accelerator()
    print(accelerator.state)

if __name__ == "__main__":
    main()
