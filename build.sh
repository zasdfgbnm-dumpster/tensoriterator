nvcc main.cu -o test -I. -std=c++17 --extended-lambda --expt-relaxed-constexpr
./test
