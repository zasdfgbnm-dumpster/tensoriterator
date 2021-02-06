nvcc add-subtract.cu -o add-subtract -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr
./add-subtract
rm -rf add add-subtract
