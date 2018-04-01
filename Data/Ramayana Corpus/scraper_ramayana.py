import requests
import pickle
from bs4 import BeautifulSoup


def f2():
	url='https://www.valmiki.iitk.ac.in/sloka'
	no=[77,119,75,67,68]
	l=[]
	for j in range(1,2):
		for k in range(1,no[j]):
			p={
				'field_kanda_tid':j,
				'language':'dv',
				'field_sarga_value':k
			}
			r=requests.get(url,params=p)
			soup=BeautifulSoup(r.text,"html5lib")

			mod=0

			for c in soup.find_all(class_="field-content"):
				if mod %3==1:
					s=c.text
					for a in s.split(','):
						x=0
						a=a.strip()
						for i in range(len(a)):
							if (a[i]>='a' and a[i]<='z') or (a[i]>='A' and a[i]<='Z'):
								x=i
								break
						l.append((a[:x-1],a[x:]))
				mod+=1
	with open("dump.pickle","wb") as p:
		pickle.dump(l)			

def f1():
	with open("output.txt","r") as f:
		x=set()
		for line in f.read():
			for c in line:
				if not((c>='a' and c<='z') or (c>='A' and c<='Z')):
					x.add(ord(c))
		print(x)
		#Range: 2306 to 2381

def main():
	f2()

if __name__ == '__main__':
	main()