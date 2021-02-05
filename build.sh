nvcc main.cu -o test -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr
./test
rm -rf test
