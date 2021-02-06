echo local
./build.sh
echo ""

echo 10.2
docker run -v $PWD:/w -w /w -it --entrypoint "/bin/bash" nvcr.io/nvidia/pytorch:19.12-py3 ./build.sh
