import os

def load_file():
    data = []
    with open("C:/WorkSpace/hw_PLA_507170627/data/train.txt", 'r') as f:
        for ff in f:
            ff = ff.replace('\n', '').split(',')
            r = [int(d) for d in ff]
            data.append(r) 
    return data

if __name__ == "__main__":
    print(load_file())