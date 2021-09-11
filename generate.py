from src.generator import Generator

if __name__ == '__main__':
    gen = Generator()
    for _ in range(100):
        print(gen.generate())

