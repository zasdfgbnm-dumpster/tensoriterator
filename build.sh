nvcc add.cu -o add -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr
./add
echo ""
# rm -rf add
