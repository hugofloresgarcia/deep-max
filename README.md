# Deep Learning for Max

you will need [min-devkit](https://github.com/Cycling74/min-devkit).


### requirements

make sure that you have libtorch installed, and that CMake can find it. 

### install

clone this repo in the `projects` folder of your min-devkit, e.g.:

```bash
# clone min-devkit
git clone https://github.com/Cycling74/min-devkit.git --recursive

cd ./min-devkit/source/projects/

git clone https://github.com/hugofloresgarcia/deep-max
```

cool! now, build min-devkit. the external will build with it. 

### Mac
From the root `min-devkit` directory, run: 

```bash
mkdir build && cd build
cmake ..
make  -j`nproc`
```
