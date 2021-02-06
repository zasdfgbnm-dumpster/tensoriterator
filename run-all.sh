echo local
./build.sh
echo ""

echo 10.2
docker run --gpus all -v $PWD:/w -w /w -it --entrypoint "/bin/bash" nvcr.io/nvidia/pytorch:19.12-py3 ./build.sh
echo ""

echo 11.0
docker run --gpus all -v $PWD:/w -w /w -it --entrypoint "/bin/bash" nvcr.io/nvidia/pytorch:20.07-py3 ./build.sh
echo ""

echo 11.1
docker run --gpus all -v $PWD:/w -w /w -it --entrypoint "/bin/bash" nvcr.io/nvidia/pytorch:20.12-py3 ./build.sh
echo ""
