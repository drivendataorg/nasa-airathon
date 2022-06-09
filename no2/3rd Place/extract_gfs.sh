cd data/airathon/raw/GFS

for city in DEL LAX TAI;
    do cd $city
    for d in PBLH RH SHFX T2 T100 U V VENT;
        do mkdir $d && tar -xvf $d.tar --directory $d/;
    done
    rm *.tar
    cd ..
done