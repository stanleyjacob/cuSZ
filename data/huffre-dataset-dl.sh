#! /bin/bash

if [[ ! -f "enwik8.zip" ]]; then
	echo "downloading enwik8"
	http://cs.fit.edu/~mmahoney/compression/enwik8.zip
else
	echo "enwik8.zip exists"
fi
if [[ ! -f "enwik8" ]]; then
	echo "decompressing enwik8"
	unzip enwik8.zip
else
	echo "enwik8 exists"
fi

if [[ ! -f "enwik9.zip" ]]; then
	echo "downloading enwik9"
	http://cs.fit.edu/~mmahoney/compression/enwik9.zip
else
	echo "enwik9.zip exists"
fi
if [[ ! -f "enwik9.zip" ]]; then
	echo "decompressing enwik9"
	unzip enwik9.zip
else
	echo "enwik9 exists"
fi

if [[ ! -f "nci.bz2" ]]; then
	echo "downloading nci"
	wget http://sun.aei.polsl.pl/~sdeor/corpus/nci.bz2
else
	echo "nci.bz2 exists"
fi
if [[ ! -f "nci" ]]; then
	echo "decompressing nci"
	bzip2 -d -k nci.bz2
else
	echo "nci exists"
fi

if [[ ! -f "mr.bz2" ]]; then
	echo "downloading mr"
	wget http://sun.aei.polsl.pl/~sdeor/corpus/mr.bz2
else
	echo "mr.bz2 exists"
fi
if [[ ! -f "mr.bz2" ]]; then
	echo "decompressing mr"
	bzip2 -d -k mr.bz2
else
	echo "mr exists"
fi

if [[ ! -f "Flan_1565.tar.gz" ]]; then
	echo "downloading Flan_1565, Rutherford Boeing format"
	wget https://suitesparse-collection-website.herokuapp.com/RB/Janna/Flan_1565.tar.gz
else
	echo "Flan_1565.tar.gz exists"
fi
if [[ ! -d Flan_1565 && ! -f Flan_1565.rb ]]; then
	echo "decompressing Flan_1565"
	tar zxf Flan_1565.tar.gz
	mv Flan_1565/* ./
	rmdir Flan_1565
elif [[ ! -f Flan_1565.rb ]]; then
	mv Flan_1565/* ./
	rmdir Flan_1565
else
	echo "Flan_1565 dir/file exists"
fi
