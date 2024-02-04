
# Download 30 weak lensing convergence maps from Hirosaki University 
# save to directory: /gpfs02/work/akira.tokiwa/gpgpu/data/WLmap_hirosaki/

for i in {0..30}
do
    # check if the file exists
    if [ -f data/WLmap_hirosaki/allskymap_nres12r$(printf "%03d" $i).zs1.mag.dat ]; then
        echo "File allskymap_nres12r$(printf "%03d" $i).zs1.mag.dat exists."
        continue
    fi
    wget http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sub1/nres12/allskymap_nres12r$(printf "%03d" $i).zs1.mag.dat -P /gpfs02/work/akira.tokiwa/gpgpu/data/WLmap_hirosaki/
    if [ -f data/WLmap_hirosaki/allskymap_nres13r$(printf "%03d" $i).zs1.mag.dat ]; then
        echo "File allskymap_nres13r$(printf "%03d" $i).zs1.mag.dat exists."
        continue
    fi
    wget http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sub1/nres13/allskymap_nres13r$(printf "%03d" $i).zs1.mag.dat -P /gpfs02/work/akira.tokiwa/gpgpu/data/WLmap_hirosaki/
done


