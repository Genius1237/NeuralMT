import os
import pickle

def clean():
    l=[]
    count=0
    for file in os.listdir('unprocessed_pickles'):
        with open('unprocessed_pickles/{}'.format(file),'rb') as f:
            p=pickle.load(f)
        for a in p:
            try:
                assert a[0].strip()!=a[1].strip()
                l.append((a[0],a[1]))
            except AssertionError:
                count+=1
    print(count)
    with open('bible_translation.pickle','wb') as f:
        pickle.dump(l,f)

def check():
    with open('bible_translation.pickle','rb') as f:
        p=pickle.load(f)
        print(len(p))
        for i in range(len(p)//5):
            try:
                print(p[5*i][0],p[5*i][1])
                print(p[5*i+1][0],p[5*i+1][1])
                print(p[5*i+2][0],p[5*i+2][1])
                print(p[5*i+3][0],p[5*i+3][1])
                print(p[5*i+4][0],p[5*i+4][1])
                input()
            except IndexError:
                break

def main():
    check()

if __name__=="__main__":
    main()