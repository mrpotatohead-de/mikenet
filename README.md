Training Acceleration with AMD BLIS
===
## Environment Setup
### Install BLIS
#### Download and Build
```bash=
#change to working directory
cd ~/ching_ws 

#clone repo from AMD official github
git clone https://github.com/flame/blis.git 

#change directory
cd ./blis

#check up environment
./configure auto

#build and install
#replace [-j] according to CPU cores on the machine
#check up cpu cores by the following line
#lscpu
make -j 6
#make [-j]
```
#### Test
```bash=
cd ~/ching_ws/blis/examples/tapi
make
./02level1m_diag.x 
```
### Install Mikenet
#### Download and Build
```bash=
#change to working directory
cd ~/ching_ws

#clone repo from our repo
#login credentials required
git clone https://aionchip.computing.ncku.edu.tw:4001/ccbl/mikenet.git --branch develop-kuo

#change directory
cd ./mikenet

#modify path in Makefile line 59
nano ./Makefile
#change path name according to where blis was installed
#for example, original =>
#lib_blis: CPPFLAGS += -fopenmp -DUSE_AMDBLIS -I/home/chingenkuo/ce_ws/blis/include/zen2
#modified =>
#lib_blis: CPPFLAGS += -fopenmp -DUSE_AMDBLIS -I/home/nm6114083/ching_ws/blis/include/zen3

#build
make clean lib
make lib_blis
```
#### Test
```bash=
cd ~/ching_ws/mikenet/demos/xor
make clean all
./xor
```
### Install OSP Model
#### Download and Build
```bash=
#change to working directory
cd ~/ching_ws

#clone repo from our repo
#download model
#login credentials required
git clone https://aionchip.computing.ncku.edu.tw:4001/ccbl/triangle-model.git --branch develop-kuo

#download headerfiles
cd ~/ching_ws/triangle-model
git clone https://aionchip.computing.ncku.edu.tw:4001/ccbl/triangle-model-headers.git

#rename folder
mv ./triangle-model-headers ./headers
```
#### Test
```bash=
export MIKENET_DIR="/home/nm6114083/ching_ws/mikenet"
cd ~/ching_ws/triangle-model/Optimization/Model
make clean raw
sh ./benchmark.sh 100
```