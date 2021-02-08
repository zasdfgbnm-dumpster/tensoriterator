nvcc add.cu -o add -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr
nvcc add-subtract.cu -o add-subtract -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr
./add
echo ""
./add-subtract
echo ""
# rm -rf add add-subtract
